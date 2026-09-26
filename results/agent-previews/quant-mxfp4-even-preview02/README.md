# Unscored gfx950 quantization agent preview

This is the raw trajectory for a **no-feedback, unscored** Codex CLI attempt
on `quant_mxfp4_even_preview` (`unscored-v1`, AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`). It is not in any
agent-error incidence denominator and does not establish AITER parity.

- Agent: Codex CLI `0.157.0`, model `gpt-5.5`, medium reasoning.
- Run: `preview02`; completed in 182.273 seconds with five captured source
  states and no snapshot errors.
- Final source: `workspace/starter.hip`, SHA256
  `a7b1dc3084ce5cd1edb0e0ea01c1730606ab59acc1439ac9e18023bd174840b5`.
- Compilation: `hipcc -O3 -shared -fPIC --offload-arch=gfx950` succeeded
  locally after the agent run. GPU correctness and performance are pending.
- Raw evidence: `events.jsonl`, `snapshots.jsonl`, `blobs/`, `diffs/`,
  `manifest.json`, `prompt.txt`, and `result.json`.
- A first invocation (`preview01`) failed before agent work because an old
  Codex authentication profile could not refresh. It is an infrastructure
  failure, not an agent kernel error. `preview02` used an existing current
  profile; no credentials are included here.

The publication check searched the copied trajectory for common API-key,
GitHub-token, private-key, and bearer-token patterns and found no matches.
The trace remains unscored even if later correctness checks find a bug.
