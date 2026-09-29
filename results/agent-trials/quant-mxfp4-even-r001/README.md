# MXFP4 Even no-feedback trial r001

This is the reviewed raw trajectory for the first
[frozen no-feedback task](../../../runs/tasks/quant_mxfp4_even_no_feedback/README.md),
run in a bubblewrap agent workspace. It is a scored-eligible **trial
capture**; independent source-trial counting is separate from repeat GPU
measurements.

- Agent: Codex CLI `0.157.0`, model `gpt-5.5`, medium reasoning, 900-second
  wall limit; it finished in 232.662 seconds with six captured source states
  and no snapshot errors. The CLI does not expose a controlled sampling seed.
- Final source: `workspace/starter.hip`, raw SHA256
  `8fe75f0e55c360edc96f5b517a46b770fe32acb7e0880348011a128d350bb8fc`;
  verified final tree SHA256
  `26d06c1b17d9fd353060e365fb472668cb994893094ba0e7cf07bf9b02c67382`.
- The agent compiled locally for `gfx950` but had no GPU device or supplied
  correctness/benchmark feedback. The trusted remote scorer compiled only
  the exported final source snapshot, outside the agent namespace.
- The [sanitized scorer summary](remote-correctness-summary.json) records a
  correctness-only pass with withheld cases evaluated. Later
  [full performance replays](../../2026-09-28-quant-r001-performance-replays.md)
  preserved the same binary and frozen task: two met the original event-only
  latency/noise rule; one failed the medium-bucket noise gate. The later
  [host-gap control](../../2026-09-29-event-timing-host-gap.md) shows those
  ratios cannot establish device-kernel parity. A revised timing protocol is
  required before any performance admission; this is not an agent-error rate.
- Raw evidence here: `events.jsonl`, `snapshots.jsonl`, `blobs/`, `diffs/`,
  `manifest.json`, `prompt.txt`, and `result.json`. Private case inputs and
  raw scorer output are not included.

The publication check searched all copied files for common API-key,
GitHub-token, private-key, bearer-token, host-home, and private-path patterns;
none contained credential material or private case values. The Codex auth
copied into the temporary agent home was readable to agent shell commands,
which is why raw event review is still required for every trial.
