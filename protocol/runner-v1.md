# Agent runner protocol v1

`runs/runner.py` records one independent HIP agent attempt per task and
replicate ID. It does not score correctness or performance. The harness must
later replay each final candidate against independent oracle and pinned AITER
on the exact same gfx950 SKU and workload. A successful CLI exit is not parity.

## Task input

Store a JSON task outside the agent workspace with these required fields:

```json
{
  "task_id": "family.variant.mode",
  "task_revision": "v1",
  "aiter_sha": "40-character-git-sha",
  "gpu_sku": "MI350X",
  "target_arch": "gfx950",
  "operator_entry": "pinned AITER entry point",
  "prompt_file": "prompt.md",
  "starter_dir": "starter",
  "harness_revision": "full harness Git SHA",
  "harness_spec_sha256": "SHA256 of pinned harness spec JSON",
  "task_mode": "no_feedback",
  "build_sources": ["kernel.hip"],
  "build_flags": ["-O3", "-shared", "-fPIC", "--offload-arch=gfx950"],
  "scored_eligible": true,
  "visible_checks": [],
  "hidden_checks": [],
  "benchmark": null
}
```

Paths are relative to the task directory. The task JSON, prompt, and starter
files are hashed by `freeze`. Only the starter directory is copied into the
agent workspace. Hidden checks and benchmark commands remain runner-side.
`no_feedback` tasks must supply no visible checks; `visible_tests_available`
tasks must list at least one. This labels *supplied* checks, not a runner-run
interactive feedback loop: an agent may choose to run them, compile, or devise
its own checks. The raw trajectory records what it actually did. Both modes
receive the same independent post-agent scoring.
The task owner must freeze public contract, hidden tests, workload buckets,
and per-bucket AITER parity thresholds before scored trials.

```sh
python3 runs/runner.py freeze tasks/example/task.json tasks/example/task.freeze.json
python3 runs/runner.py run tasks/example/task.json \
  --freeze tasks/example/task.freeze.json --replicate-id r001 \
  --model gpt-6-sol --reasoning-effort medium --wall-seconds 1800 --dry-run
```

Remove `--dry-run` only after task, harness, GPU access, and baseline have
been verified. The runner fixes wall time, CLI version, model, reasoning effort,
prompt hash,
task/harness revision, AITER SHA, SKU, and workspace starter hash. Codex CLI
does not expose a sampling seed in this invocation; `replicate_id` identifies
independent sessions and must not be interpreted as a controlled seed.
An actual agent launch requires `scored_eligible: true`, a full pinned harness
SHA, spec hash, and frozen build command; the dry-run fixture deliberately
fails that gate.
For exploratory implementation before task admission, `--unscored-preview`
actually invokes Codex on a task with `scored_eligible: false` and retains all
snapshots. Its manifest records `incidence_eligible: false`, and `score`
refuses that run. It must never enter an agent-error frequency denominator.
The current runner also has no reliable model-token cap, so trials needing
strict token-budget equality are not scored until that control is added.

## Agent isolation

Every real `run`, including an unscored preview, requires unprivileged
`bwrap`; there is no unsandboxed fallback. The agent sees only its writable
`/workspace`, read-only `/usr` and `/etc`, the selected read-only ROCm
toolchain, a single resolver file if needed, the Codex executable, and a
configured CA certificate. `/tmp` and `/home/agent/.codex` are private tmpfs
mounts. The trusted runner copies Codex auth and policy cache files into that
tmpfs by file descriptor; refreshed state is not written back to the host.
PID, IPC, and UTS namespaces are unshared. The basic `/dev` mount does not
include `/dev/kfd` or `/dev/dri`, so agent-side GPU execution is unavailable;
`hipcc` compilation still works. Network is shared for the model API.

The host home, SSH keys, SSH agent socket, AITER checkout, private withheld
manifest, harness, and unrelated workspace paths are not mounted. The runner
clears inherited environment variables, so SSH and cloud credentials are not
forwarded. The post-agent scorer runs separately, after the sandbox exits.
The boundary test verifies the hidden paths and credential variables are
absent while workspace writes and `hipcc` work. A real unscored Codex smoke
also completed in this namespace with `CODEX_HOME` pointing to a current
host auth profile (21 structured events, empty stderr). A manual SSH probe
from the namespace to the target host's direct address was denied with
`Permission denied (publickey)`; the host's `mi350-2` SSH alias is absent
because the host SSH config is not mounted.

This is filesystem/credential isolation for the study, not a defense against
a malicious model with arbitrary network access: the Codex process and its
shell tools share the namespace, so agent commands **can read the copied
Codex auth JSON**. The boundary test records that fact without recording any
token bytes. It protects host SSH credentials and withheld data, not the
model API credential itself. Use a dedicated, limited credential for larger
studies and keep the raw event stream private until a secret review. No agent-side GPU
feedback is available under this policy; any task advertised as having
visible GPU benchmarks must be relabeled or given a separately controlled
feedback channel before scored enrollment.

## Artifacts and limitations

Each actual run writes to `runs/artifacts/<task>--<revision>--<replicate>/`:
the manifest, exact prompt, raw Codex JSONL events, stderr, content-addressed
source blobs, ordered tree snapshots, diffs, and final process result. The
runner polls source files and snapshots at agent completion events. It cannot
guarantee capture of a transient edit that exists only between two polls;
source changes made by an uninstrumented external process are also outside
the completeness claim. Candidate source is never overwritten in the blob
store. Limits on file size/count fail the run visibly rather than silently
dropping source.

Raw artifacts are gitignored because agent prompts and tool outputs can
contain credentials or private data. Review and redact them before publishing
results. The agent workspace is separated from task/harness paths for
accidental-leak resistance, not a security boundary against a malicious
agent; scored trials need host/container isolation of hidden tests.

After the agent exits, score from a trusted control plane:

```sh
python3 runs/runner.py score --task tasks/example/task.json \
  --freeze tasks/example/task.freeze.json --run-dir runs/artifacts/<run-id> \
  --harness-root <clean-pinned-harness-checkout> \
  --spec <clean-pinned-harness-checkout>/references/<spec>.json \
  --aiter-source <clean-pinned-AITER-checkout> \
  --withheld-spec <private-runner-only-cases.json> \
  --host-gpu-report <private-host-ROCm-report.json> \
  --output runs/scores/<run-id> --snapshot all --correctness-only
```

`score` verifies the task freeze, pinned clean Git checkouts, spec hash, and
the raw SHA256 commitment from the public spec to the private withheld-case
manifest. It also checks that the host report identifies the frozen gfx950 SKU
and PCI model; both private files must remain outside the agent run directory,
and the withheld manifest must remain outside the public harness checkout.
Private file paths are redacted from score records; only their hashes are
retained there. Never publish the raw harness result or scorer logs before
review: the raw result contains withheld case IDs.

The runner verifies each snapshot/blob hash. It exports sources to a new
directory, compiles with
the frozen `hipcc` argv into `libcandidate.so`, and invokes the scorer on that
binary. It never invokes an agent-written build script or passes the live
workspace to the scorer. The scorer must enforce correctness before timing;
the runner validates the returned task/AITER/spec/binary/source hashes and
the withheld and host-report hashes. A wrong candidate can yield a valid
scorer result with exit code 1; the runner records that as a completed score
with `joint_pass: false`, not an infrastructure failure. Use `--snapshot
final` for just the submitted source and omit `--correctness-only` only when
the GPU is uncontended and the benchmark contract is admitted.

The scorer consumes the native C ABI in
`references/quant_mxfp4_abi.h`: `aiter_rs_quant_mxfp4_even` accepts GPU input,
packed/scales outputs, dimensions, dtype, and a HIP stream, returning a HIP
error code. Its `result.json` stores correctness and per-bucket performance,
and hashes the compiled `.so` separately from the source snapshot. The
example under `runs/examples/` is a **no-score** CLI preview, not a validated
kernel task.
