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
- Three additional [GDR native-seed optimization previews](agent-trials/gdr-native-optimization-previews.md)
  were captured with one fixed CLI/model/effort and no GPU feedback. Trusted
  [exploratory replay](agent-trials/gdr-native-optimization-replay-20260929.md)
  found all three functionally correct on 6/6 visible plus 4/4 withheld cases,
  but all missed at least one public graph bucket's <=1.05 latency gate.
  Preview01/02 slowed all three buckets; preview03 slowed only batch 16
  (1.0623x, then 1.0725x in a fresh repeat). These are agent-authored
  performance regressions on one optimization task, but **unscored previews**
  outside the incidence denominator, with no isolated source-level cost cause.

## Analyst and AITER controls

| Track | Current evidence | Admission status |
| --- | --- | --- |
| GDR decode | Naive HIP passed public correctness but was 5.76-5.85x slower under repeatable graph replay. A source-preserving AITER-derived HIP ABI seed passed 8/8 public+withheld correctness. [Profiler and stability controls](../tasks/gdr_native_optimization/PROFILER_STABILITY_RECEIPT.md) confirmed one same-symbol, same-geometry dispatch per side in three public buckets and two fresh graph sessions within 5% of AITER. The [public-only scorer smoke](../runs/tasks/gdr_native_optimization_v1/PUBLIC_SEED_SMOKE.md) passed six visible checks and three parity buckets. The later host-owned replay seed passed all public cases and buckets in its separate environment. | Credible HIP parity seed and three exploratory no-feedback agent replays. A GDR-specific post-agent hidden replay exists, but no live agent feedback broker, frozen scored environment, or scored agent attempt. The seed is an answer key for optimization, not blind from-spec trials. |
| MHC fused post/pre | Guarded analyst HIP passed 9/9 cases, but graph replay was 20-106x slower than AITER. | No credible HIP parity seed; unscored. |
| Sparse prefill | Analyst HIP passed 5 public and 4 withheld correctness cases. Read-only post-graph verification passed, but two public graph buckets were 9.97x and 11.86x slower. | No credible HIP parity seed; unscored. |
| OPUS persistent A16W16 GEMM | Three original exact-kid cases passed the independent Torch oracle and the [source-equivalent standalone HIP adapter](../tasks/opus_a16w16_persistent/FEASIBILITY.md). Eight full-K public cases then passed guarded multi-step correctness and one-session [graph control](../tasks/opus_a16w16_persistent/ADVERSARIAL_GRAPH_FEASIBILITY.md) at 0.9953-1.0244 adapter/AITER. A random-input partial-K pretrial exposed an AITER-only [padding-dependent correctness defect](../tasks/opus_a16w16_persistent/K_TAIL_REPRO.md), independently reproduced and filed as [ROCm/aiter #5954](https://github.com/ROCm/aiter/issues/5954). A stream-zero adapter bug was caught before launch; neither it nor the AITER finding is a scored agent mistake. | Plausible source-preserving route for the eight full-K cases only. Partial-K excluded pending upstream fix and re-admission. No hidden matrix, frozen agent optimization task, or agent trajectory. |
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

Next: connect the GDR public broker and freeze a scored environment/private
replay protocol before any **scored** attempt; the three exploratory captures
remain outside the incidence denominator. For megakernels, freeze a hidden
matrix and clean-JIT source-overlay task for the eight supported full-K OPUS
persistent cases, with a second independent performance session. The
standalone adapter is only a source-equivalent control.
Keep FlyDSL-only and binary-only megakernel paths outside source-level HIP
agent-error incidence until a legitimate same-boundary route exists.
