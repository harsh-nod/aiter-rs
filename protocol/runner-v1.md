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
  "scored_eligible": true,
  "visible_checks": [["bash", "visible.sh"]],
  "hidden_checks": [],
  "benchmark": null
}
```

Paths are relative to the task directory. The task JSON, prompt, and starter
files are hashed by `freeze`. Only the starter directory is copied into the
agent workspace. Hidden checks and benchmark commands remain runner-side.
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
An actual agent launch requires `scored_eligible: true` and a full pinned
harness SHA; the dry-run fixture deliberately fails that gate.
The current runner also has no reliable model-token cap, so trials needing
strict token-budget equality are not scored until that control is added.

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

Only correctness-passing candidates may be benchmarked. Neither this runner
nor `codex exec` provides that judgment. Harness results should reference
`task_freeze_sha256`, `final_tree_sha256`, exact hardware/environment, and
per-bucket AITER baseline revision so failures and parity can be replayed.
The expected handoff is `python3 -m harness.run --spec <json> --candidate
<candidate-entry-or-dir> --aiter-source <pinned-AITER-clone> --output <new-dir>
--task-freeze-sha256 <sha> --final-tree-sha256 <sha>`. The harness records its
own candidate-path hash separately from this runner's source-tree hash. The
example under
`runs/examples/` is a **no-score** CLI preview, not a task with a validated
contract or oracle.
