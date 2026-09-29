# GDR live public-feedback prototype

This is a new, **unscored** task revision. It copies the v1 native HIP seed and
public contract, but changes the agent protocol from no feedback to a live,
host-brokered public-only request/response path. Old v1 captures remain frozen
and are not relabeled. The task remains `scored_eligible=false` until the live
broker and later private replay pass admission; no attempt from this revision
belongs in an agent-error incidence denominator yet.

The agent sees a read-only helper at `/public-feedback.py` and can request at
most three public checks. The isolated runner mounts only that helper and its
workspace; the trusted host owns SSH credentials, immutable source snapshots,
remote Docker execution, and raw scorer results. Requests must name the SHA256
of the current `kernel.hip`. Responses contain only allowlisted visible case
booleans, public graph bucket ratios/pass flags, and hashes. The remote public
container mounts no withheld manifest, has no network, and runs as the
non-root host UID with a minimal read-only NSS entry, `HOME=/tmp`, and a
writable `/tmp/aiter-jit-cache`.

The runner's Codex model API network remains enabled, and Codex auth is copied
into an ephemeral namespace readable by agent shell commands. That is not
total credential isolation. SSH credentials and the withheld manifest are not
mounted in the agent sandbox. A later private scorer still loads native
candidate code in the same process as the withheld manifest, so it does not
provide strong confidentiality against malicious native submissions.

Admission sequence: CPU fake end-to-end broker tests; one bounded MI350X
public-only smoke with pinned host/container checks; a separate private
post-agent correctness replay; only then consider a new scored-eligible
revision. Do not change this task's eligibility flag in place.

## Trusted operator route

The remote host needs a clean checkout containing this revision, pinned AITER,
the pinned MI350X host report, and an empty private root. The config is a
host-only JSON file outside the repository and capture tree with exactly these
fields:

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
  runs/tasks/gdr_native_optimization_v2/task.json \
  --freeze runs/tasks/gdr_native_optimization_v2/task.freeze.json \
  --results /private/gdr-live-captures --replicate-id <fresh-id> \
  --model <fixed-model> --reasoning-effort <fixed-effort> \
  --wall-seconds 900 --unscored-preview \
  --feedback-config /private/gdr-live-public-config.json \
  --feedback-private-root /private/gdr-live-feedback-raw
```

The agent can call `python3 /public-feedback.py correctness` or `benchmark`.
Each call first freezes the current source, then executes the pinned public
scorer over SSH in an independently attested, network-disabled Docker stage.
The host-only journal commits request, source, snapshot, response, and raw
result hashes. All remote stdout/stderr and raw results remain private. A
failed request consumes one of the three slots. A timeout is an infrastructure
outcome, not a kernel correctness finding.

Only after the agent exits, `runs.gdr_native_optimization.post_agent` can
replay a completed v2 capture using `--task-dir` pointed to this revision and
the separate private withheld manifest. That replay is exploratory and
incidence-ineligible. CPU fake captures test protocol behavior only; they are
not evidence that an agent made or repaired a GPU-kernel bug.
