# Agent-kernel study: 2026-09-29 progress ledger

This is a research status snapshot, not a success-rate estimate. All reported
device work ran on `mi350-2` (MI350X/gfx950) against pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. Private matrices, raw
host reports, and raw GPU results remain off-repository; public receipts
contain commitments and sanitized aggregates.

## Agent-induced evidence

- Six independent, frozen HIP captures on **one** MXFP4 Even task, not six
  kernel types: r001/r004/r005 passed final correctness 7/7; r002/r003/r006
  passed only 1/7. All three failures were silent packed-output corruption
  involving a subgroup source-lane participation pattern. Separate one-line
  analyst controls changed r002 and r006 from 1/7 to 7/7; r003 has a
  within-trajectory correct shared-memory snapshot followed by a failing
  shuffle rewrite, but no one-line isolation. See the
  [case study](../analysis/quant-wave-participation.md).
- These were no-feedback runs on one contract. The medium and high tranches
  also used different Codex CLI versions, so they cannot isolate a
  reasoning-effort effect. No population incidence, megakernel incidence, or
  cross-task frequency follows from 3/6.
- Earlier direct GPU-event timing included unequal host enqueue gaps. A
  [graph control](../references/quant_mxfp4_graph_control.md) found correct
  r001 slower than AITER in both public buckets (1.35x and 1.96x). None of
  the six runs has established device-performance parity.

## Analyst and AITER controls

| Track | Current evidence | Admission status |
| --- | --- | --- |
| GDR decode | Naive HIP passed public correctness but was 5.76-5.85x slower under repeatable graph replay. A source-preserving AITER-derived HIP ABI seed passed 8/8 public+withheld correctness and two graph runs near 1.00x. A proposed batch-16 fixture also passed two three-bucket graph controls. | Credible HIP parity route, but the [optimization freeze](../tasks/gdr_native_optimization/FREEZE_DRAFT.md) remains a draft pending profiler, stability, and agent/private-data isolation gates. The published seed is an answer key for optimization, not blind from-spec trials. |
| MHC fused post/pre | Guarded analyst HIP passed 9/9 cases, but graph replay was 20-106x slower than AITER. | No credible HIP parity seed; unscored. |
| Sparse prefill | Analyst HIP passed 5 public and 4 withheld correctness cases. Read-only post-graph verification passed, but two public graph buckets were 9.97x and 11.86x slower. | No credible HIP parity seed; unscored. |
| OPUS persistent A16W16 GEMM | Three supported exact-kid gfx950 cases passed an independent Torch oracle twice; profiler confirmed persistent kernel symbols for kids 300 and 1300. An `N=2177` pretrial was excluded by the pinned `N % 16 == 0` tuner domain. | Correctness/dispatch only; no standalone HIP candidate, hidden matrix, or parity timing. |
| TopK long-row | Pinned FlyDSL route matched an independent oracle on 20 calls. | Not a HIP pilot without a parity adapter. |

The separate AITER GDR `scale=NaN` validation bypass is tracked in the
[audit receipt](2026-09-29-gdr-scale-nan.md) and upstream issue #5951. It is
an upstream-wrapper finding, **not** an agent-written-kernel mistake. The
[fe2o3 IR control](2026-09-29-fe2o3-wave-ir-gap.md) rejects an explicit
partial-lane collective claim but accepts a false full-wave claim inside a
lane-divergent branch; this is not an end-to-end Rust-source proof.

## Scale and next gates

The initial 36-trajectory pilot has six captures, all on one task. The
[gfx950 census](../inventory/results_census.md) contains source-entry and
configuration hints, not 10,000 distinct vetted HIP challenge types. Do not
inflate type count with shape rows, repeated attempts, or AITER-only audits.

Next: freeze and isolate the GDR native-seed optimization task with a
profiler-backed performance boundary and realistic visible feedback; run
independent agent attempts and preserve every failed intermediate snapshot.
For megakernels, advance the exact-kid persistent GEMM from dispatch admission
to a standalone HIP parity seed and hidden adversarial matrix. Keep the
FlyDSL-only and binary-only megakernel paths outside source-level HIP
agent-error incidence until a legitimate same-boundary route exists.
