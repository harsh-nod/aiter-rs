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
| GDR decode | Naive HIP passed public correctness but was 5.76-5.85x slower under repeatable graph replay. A source-preserving AITER-derived HIP ABI seed passed 8/8 public+withheld correctness. [Profiler and stability controls](../tasks/gdr_native_optimization/PROFILER_STABILITY_RECEIPT.md) confirmed one same-symbol, same-geometry dispatch per side in three public buckets and two fresh graph sessions within 5% of AITER. The [public-only scorer smoke](../runs/tasks/gdr_native_optimization_v1/PUBLIC_SEED_SMOKE.md) passed six visible checks and three parity buckets; its geometric-mean ratio 1.0112 missed the proposed 0.95 improvement objective. | Credible HIP parity seed for an unscored, no-feedback optimization prototype. No live agent feedback broker, GDR-specific trusted post-agent hidden replay, or scored agent attempt yet. The seed is an answer key for optimization, not blind from-spec trials. |
| MHC fused post/pre | Guarded analyst HIP passed 9/9 cases, but graph replay was 20-106x slower than AITER. | No credible HIP parity seed; unscored. |
| Sparse prefill | Analyst HIP passed 5 public and 4 withheld correctness cases. Read-only post-graph verification passed, but two public graph buckets were 9.97x and 11.86x slower. | No credible HIP parity seed; unscored. |
| OPUS persistent A16W16 GEMM | Three supported exact-kid gfx950 cases passed an independent Torch oracle twice; profiler confirmed persistent kernel symbols for kids 300 and 1300. A [source-equivalent standalone HIP adapter](../tasks/opus_a16w16_persistent/FEASIBILITY.md) passed the same three cases twice, was bitwise identical to AITER, and measured 0.9934/1.0001/0.9787 adapter/AITER in 32-call graph replay. A stream-zero adapter bug was caught before launch and fixed; it is not a scored agent mistake. An `N=2177` pretrial was excluded by the pinned `N % 16 == 0` tuner domain. | Plausible source-preserving parity route, but no hidden adversarial matrix, frozen agent optimization task, or agent trajectory. The adapter is an analyst control, not an independent rewrite. |
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

Next: connect the GDR public broker and a GDR-specific trusted post-agent
hidden replay before any **scored** attempt. Exploratory no-feedback captures
can preserve failed intermediate snapshots, but remain outside the incidence
denominator until a scored protocol is frozen. For megakernels, build a hidden
adversarial matrix and clean-JIT source-overlay task for the exact-kid OPUS
persistent GEMM; the standalone adapter is only a source-equivalent control.
Keep FlyDSL-only and binary-only megakernel paths outside source-level HIP
agent-error incidence until a legitimate same-boundary route exists.
