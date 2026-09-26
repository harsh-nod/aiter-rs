# MXFP4 Even no-feedback trial r003

This is the reviewed raw trajectory for a third independent run of the
[frozen no-feedback task](../../../runs/tasks/quant_mxfp4_even_no_feedback/README.md).
It used a fresh bubblewrap workspace and the same public contract, private
case commitment, model, and wall limit as r001/r002. The agent submission is
preserved unchanged.

- Agent: Codex CLI `0.157.0`, model `gpt-5.5`, medium reasoning, 900-second
  wall limit. It finished in 178.231 seconds with three captured source states
  and no snapshot errors. The CLI does not expose a controlled sampling seed.
- Final source: `workspace/starter.hip`, raw SHA256
  `3e2186dfa2431bb02315494099728691f96f3068d7c8849217f257354c56dea1`;
  final tree SHA256
  `c7ece69794bd4a79b9f73e629308f95113f06c36a28b085351672bb52ab46f22`.
- The [sanitized final score](remote-correctness-summary.json) records a
  trusted correctness-only replay: pinned AITER passed 7/7 cases, while the
  submitted candidate passed 1/7. Six cases had packed-output mismatches;
  scale outputs and input integrity passed. Performance was not run.
- The agent's first compiled source (snapshot 2) used shared-memory reduction
  and packing. Its [separate historical replay](snapshot2-analyst-control-summary.json)
  passed 7/7, including withheld cases. The agent then rewrote the kernel to
  use wave shuffles for speed; the final code calls `__shfl` inside a
  `lane < 16` branch while requesting values from lanes 16-31. The failure
  appeared during this agent optimization. The snapshot-2 replay is an
  analyst control, not another independent trial or final submission.
- Raw evidence here: `events.jsonl`, `snapshots.jsonl`, `blobs/`, `diffs/`,
  `manifest.json`, `prompt.txt`, and `result.json`. Private case inputs and
  raw scorer output are not included.

The publication check searched all copied files for common API-key,
GitHub-token, private-key, bearer-token, host-home, and private-path patterns;
none contained credential material or private case values. The Codex auth
copied into the temporary agent home was readable to agent shell commands,
so raw event review remains required.
