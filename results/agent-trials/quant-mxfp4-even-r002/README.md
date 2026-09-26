# MXFP4 Even no-feedback trial r002

This is the reviewed raw trajectory for an independent repeat of the
[frozen no-feedback task](../../../runs/tasks/quant_mxfp4_even_no_feedback/README.md).
It used a fresh bubblewrap workspace and the same public contract, private
case commitment, model, and wall limit as r001. The agent submission is
preserved unchanged.

- Agent: Codex CLI `0.157.0`, model `gpt-5.5`, medium reasoning, 900-second
  wall limit; it finished in 187.098 seconds with three captured source states
  and no snapshot errors. The CLI does not expose a controlled sampling seed.
- Final source: `workspace/starter.hip`, raw SHA256
  `da218f93007fc648a12ea9c0517a508f40e93e2ef36bcea73a69202c491514f0`;
  verified final tree SHA256
  `a4e604643622796067a8a55cb88ff3dc2352205371ac36ece7dead7d6a802581`.
- The [sanitized scorer summary](remote-correctness-summary.json) records a
  trusted correctness-only replay: pinned AITER passed 7/7 cases, while the
  candidate passed 1/7. Its failures are packed-output mismatches; scale
  outputs and input integrity passed. Performance was not run.
- The candidate source places `__shfl_down(code, 1)` inside an even-lane-only
  branch, the same source-lane participation hazard confirmed by the
  [preview 03 controlled repair](../../2026-09-25-quant-mxfp4-preview03-visible.md).
  A separate repair check for r002 is recorded only if independently run;
  the agent's own source remains the failure evidence.
- Raw evidence here: `events.jsonl`, `snapshots.jsonl`, `blobs/`, `diffs/`,
  `manifest.json`, `prompt.txt`, and `result.json`. Private case inputs and
  raw scorer output are not included.

The publication check searched all copied files for common API-key,
GitHub-token, private-key, bearer-token, host-home, and private-path patterns;
none contained credential material or private case values. The Codex auth
copied into the temporary agent home was readable to agent shell commands,
so raw event review remains required.
