# aiter-rs

Agent-first study of mistakes introduced while writing or optimizing
gfx950 AITER workloads in HIP, especially megakernels, and the practical
value of Rust-based verification with
[fe2o3](https://github.com/powderluv/fe2o3).

Agents will implement HIP kernels from specifications or optimize vetted HIP
baselines while we retain all candidate changes and test them independently.
Joint success requires matching the pinned AITER gfx950 operator's functionality
and performance on the same GPU and workload; failed attempts remain in the
error corpus. Correctness also uses an independent oracle. After a pilot, the
study aims for 10,000 meaningfully distinct AITER-grounded challenge types,
each with an independent agent run, subject to a source-inventory and resource
feasibility gate. A separate AITER source audit supplies validation cases,
but verifier priorities come from observed, transferable agent failures.
See [PLAN.md](PLAN.md) for the protocol and parallel workstreams.

## Current state (2026-09-25)

- [The selected gfx950 registry](inventory/dispatch_tranche.md) traces six
  AITER families and 12 provisional algorithm/contract regimes. This is not
  a complete census; it predates the first frozen quant trial below. No task
  has a completed joint-parity result, and there is no evidence-backed route
  to 10,000 distinct types yet.
- The first [MXFP4 Even task](tasks/quant_mxfp4_even/spec.md) has an
  independent oracle and a HIP reference. A
  [correctness check](results/2026-09-25-quant-mxfp4-baseline-admission.md)
  passed all seven cases, including four withheld, on one MI350X. Performance
  parity has not been established.
- Two unscored Codex HIP trajectories are preserved with source snapshots
  and raw events: [preview 02](results/agent-previews/quant-mxfp4-even-preview02/README.md)
  passed all seven correctness cases in trusted replay, while [preview 03](results/agent-previews/quant-mxfp4-even-preview03/README.md)
  exposed a reproducible divergent wave-shuffle bug. Neither enters an
  agent-error incidence denominator. The failure's
  [fe2o3 mapping](results/2026-09-25-fe2o3-wave-shuffle-target.md) is a
  proposed verifier challenge, not a claim of current detection.
- Two independent frozen no-feedback quant trials have trusted correctness
  replays: [r001](results/agent-trials/quant-mxfp4-even-r001/README.md)
  passed 7/7 cases, while [r002](results/agent-trials/quant-mxfp4-even-r002/README.md)
  passed 1/7. The r002 agent placed a wave shuffle inside an even-lane branch;
  a separate [one-line analyst control](results/agent-trials/quant-mxfp4-even-r002/analyst-shuffle-repair-summary.json)
  passed 7/7 and supports that failure mechanism. The repair is not an agent
  trial. Neither trial has a performance or joint-parity outcome yet.
- The [GDR decode task](references/gdr_decode_packed_bf16_contract.md) has
  an independent stateful CPU oracle. Pinned AITER passed its
  [eight-case correctness matrix](results/2026-09-25-gdr-decode-bf16-baseline-admission.md);
  a [guarded HIP candidate scorer](results/2026-09-25-gdr-candidate-scorer-smoke.md)
  now exists, but no functional agent HIP candidate or performance parity has
  been established.
- [MegaMoE feasibility](mega/feasibility.md) documents the cross-rank
  hardware gap: the current `mi350-2` session exposes one GPU, so a
  world-size-one test cannot validate its multi-GPU protocol. The visible
  GPU is occupied by an unrelated workload; no trustworthy latency parity
  measurement has been run.

The pinned AITER checkout remains outside this repository. Public agent
artifacts are reviewed for credentials and private data before publication;
the withheld test matrix stays private. The runner and harness are still
pilot infrastructure, not a completed 10,000-task study.
