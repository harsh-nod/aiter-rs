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
  a complete census, and **zero types are admitted for scored trials**. There
  is no evidence-backed route to 10,000 distinct types yet.
- The first [MXFP4 Even task](tasks/quant_mxfp4_even/spec.md) has an
  independent oracle and a HIP reference. A
  [correctness check](results/2026-09-25-quant-mxfp4-baseline-admission.md)
  passed all seven cases, including four withheld, on one MI350X. Performance
  parity has not been established.
- One [unscored Codex HIP trajectory](results/agent-previews/quant-mxfp4-even-preview02/README.md)
  is preserved with source snapshots and raw events. Its
  [visible-only check](results/2026-09-25-quant-mxfp4-preview02-visible.md)
  passed three cases. It is exploratory and must not enter an agent-error
  incidence denominator.
- [MegaMoE feasibility](mega/feasibility.md) documents the cross-rank
  hardware gap: the current `mi350-2` session exposes one GPU, so a
  world-size-one test cannot validate its multi-GPU protocol. The visible
  GPU is occupied by an unrelated workload; no trustworthy latency parity
  measurement has been run.

The pinned AITER checkout remains outside this repository. Public agent
artifacts are reviewed for credentials and private data before publication;
the withheld test matrix stays private. The runner and harness are still
pilot infrastructure, not a completed 10,000-task study.
