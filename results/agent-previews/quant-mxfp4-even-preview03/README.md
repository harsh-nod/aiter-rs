# Isolated gfx950 quantization agent preview 03

This is the raw trajectory of a **no-feedback, unscored** Codex CLI attempt
on `quant_mxfp4_even_preview` (`unscored-v1`, AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`). It is not in the
agent-error incidence denominator. The independent
[visible-only result](../../2026-09-25-quant-mxfp4-preview03-visible.md)
reproduces a divergent wave-shuffle error; no hidden case or latency was run.

- Agent: Codex CLI `0.157.0`, model `gpt-5.5`, medium reasoning.
- Run: `preview03`; completed in 263.731 seconds with five captured source
  states and no snapshot errors.
- Final source: `workspace/starter.hip`, SHA256
  `1a6db9ab07e94a2973e61a9e1275bd274c699a3cd1420d7ed34f8b065b6194d2`.
- Isolation: bubblewrap filesystem namespace; no host SSH credentials,
  private test area, or GPU devices in the agent workspace. The Codex auth
  copied into its temporary home was readable to agent commands; raw events
  were reviewed before publication.
- Raw evidence: `events.jsonl`, `snapshots.jsonl`, `blobs/`, `diffs/`,
  `manifest.json`, `prompt.txt`, and `result.json`.

The publication check searched all copied files for common API-key,
GitHub-token, private-key, bearer-token, host-home, and private-path patterns.
No credential material or private case values were found. This run remains
exploratory even though its error mechanism is reproducible.
