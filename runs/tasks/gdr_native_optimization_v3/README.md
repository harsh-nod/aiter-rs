# GDR v3 Scored Live-Feedback Task

This is a new scored-protocol candidate revision of the native HIP GDR
optimization task. Its task metadata declares `scored_eligible=true`, but the
runner has a fail-closed launch gate: no agent session can start until a
separate, committed `admission.json` records reviewed exact-v3 public and
withheld smoke result hashes against this freeze and the pinned environment.
The v1 no-feedback and v2 unscored live-feedback tasks and their captures remain
unchanged. Eligibility is a protocol designation, not evidence that an agent
has passed correctness or performance. No v3 agent session should launch until
the integrated freeze, broker, and trusted public/withheld replay are admitted
on the pinned MI350X environment.

An agent may make at most three live public-only requests, each bound to a
source SHA256 and immutable snapshot. The broker returns only allowlisted
visible correctness booleans, public graph bucket ratios/pass flags, and
hashes. The final immutable source snapshot, not the best feedback request,
is the scored candidate. Withheld inputs are only used after the agent exits.
The remote public container has no withheld mount or network, uses the host
non-root UID, a minimal read-only NSS entry, `HOME=/tmp`, and a writable
`/tmp/aiter-jit-cache`. The task pins the AITER SHA, public/withheld
commitments, host report, image, compiler, gfx950 target, and scorer revision.

The correctness gate includes all visible and withheld cases. Each of the
three graph buckets must be noise-qualified on both sides (relative MAD at
most 0.05) and have candidate/AITER latency ratio at most 1.05. A separate
improvement objective requires the ratio geomean at most 0.95 after parity.
These are distinct outcomes; a parity pass does not imply an improvement.

Every launched v3 agent session belongs in the incidence denominator,
including no-feedback runs, agent timeouts, nonzero exits, compile failures,
and correctness failures. Capture and trusted score are separate records.
A private `agent-launch.json` is written after process creation; the trusted
`capture_incidence_record` uses that receipt, rather than a prelaunch manifest
alone, to identify denominator members. It still counts an interrupted run
with no `result.json`.
A failed or incomplete capture remains a denominator entry even when it cannot
be replayed. Infrastructure-invalid scoring retains the entry and requires a
predeclared retry of the same final snapshot. Public or withheld watchdog
timeouts are ambiguous between a candidate hang and an environment failure:
retain the entry, inspect the private preflight/PID/log evidence and reproduce
before assigning blame or allowing a retry. Never silently drop a run or count
an unscored v1/v2 preview in the v3 denominator.

The runner's model API network remains enabled and its ephemeral Codex auth is
readable to agent shell commands. SSH credentials and the withheld manifest
are not mounted in the agent sandbox. The private scorer loads candidate
native code in the same process as hidden inputs. Thus the design protects
against accidental disclosure, not adversarial exfiltration by a malicious
candidate. Keep raw logs, private paths, and case data outside the repository.

## Trusted Operator Route

The separate, reviewed `admission.json` is intentionally absent from this
candidate freeze. Its schema is `aiter-rs-gdr-v3-admission-v1`; it must contain
`status=admitted`, this task revision and freeze SHA256, pinned AITER SHA,
host-report SHA256, compiler image ID, and SHA256 hashes of exact-v3 public
and withheld smoke results. The runner requires it to be committed and clean.
Neither v2 smoke nor a local fake test satisfies this gate.

The remote host needs a clean checkout containing this revision, pinned AITER,
the pinned MI350X host report, and an empty private root. The host-only config
outside the repository and capture tree has exactly these fields:

```json
{
  "schema": "aiter-rs-gdr-public-ssh-v1",
  "ssh_host": "mi350-2",
  "remote_root": "/private/gdr-live-public",
  "remote_repo": "/trusted/aiter-rs",
  "remote_aiter": "/trusted/aiter",
  "remote_host_gpu_report": "/private/mi350x-host-report.json",
  "remote_repo_head": "<full pinned Git SHA>",
  "image": "<pinned local ROCm image name>",
  "public_wall_seconds": 300
}
```

With that config and the task's freeze, the local capture command is:

```bash
CODEX_HOME=/home/harsh/.codex-openai python3 -m runs.runner run \
  runs/tasks/gdr_native_optimization_v3/task.json \
  --freeze runs/tasks/gdr_native_optimization_v3/task.freeze.json \
  --results /private/gdr-live-captures --replicate-id <fresh-id> \
  --model <fixed-model> --reasoning-effort <fixed-effort> \
  --wall-seconds 900 \
  --feedback-config /private/gdr-live-public-config.json \
  --feedback-private-root /private/gdr-live-feedback-raw
```

The agent can call `python3 /public-feedback.py correctness` or `benchmark`.
Each call first freezes the current source, then executes the pinned public
scorer over SSH in an independently attested, network-disabled Docker stage.
The host-only journal commits request, source, snapshot, response, and raw
result hashes. All remote stdout/stderr and raw results remain private. A
failed request consumes one of the three slots. A live public request timeout
is candidate-versus-environment ambiguous. Its private evidence and source
snapshot remain attached to the run; it is not automatically retried or
charged as a kernel fault. The final snapshot can still receive a separate
post-agent trusted score after a clean capture.

Only after the agent exits, `runs.gdr_native_optimization.post_agent` can
replay a completed v3 capture using `--task-dir` pointed to this revision and
the separate private withheld manifest. CPU fake captures test protocol
behavior only; they are not agent evidence or scored trials.
