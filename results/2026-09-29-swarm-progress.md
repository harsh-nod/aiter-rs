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
- A separate [GDR live-feedback preview](agent-trials/gdr-native-live-feedback-preview-001.md)
  used three agent-visible public benchmark requests. All tested source
  snapshots passed visible correctness; the third had two noise-qualified
  latency-ratio misses (1.0508 and 1.0621). The agent then submitted a
  distinct, untested final source. Post-agent replay passed 6/6 public and
  4/4 withheld correctness and all three public non-inferiority buckets, but
  missed the separate 0.95 geomean improvement target (ratio 1.007347).
  This is **one unscored preview**, not an incidence observation or a new type.

## Analyst and AITER controls

| Track | Current evidence | Admission status |
| --- | --- | --- |
| GDR decode | Naive HIP passed public correctness but was 5.76-5.85x slower under repeatable graph replay. A source-preserving AITER-derived HIP ABI seed passed 8/8 public+withheld correctness. [Profiler and stability controls](../tasks/gdr_native_optimization/PROFILER_STABILITY_RECEIPT.md) confirmed one same-symbol, same-geometry dispatch per side in three public buckets and two fresh graph sessions within 5% of AITER. The [public-only live-broker seed smoke](2026-09-29-gdr-live-feedback-admission.md) passed 6/6 visible checks, and one real live-feedback agent preview completed with separate public and withheld replay. | Credible HIP parity seed; three exploratory no-feedback and one live-feedback agent preview. The [v3 scored-protocol candidate](../runs/tasks/gdr_native_optimization_v3/README.md) now freezes the live task and nonadversarial same-process hidden-scorer limitation, with all launched sessions in the denominator. Its admission receipt is intentionally absent: exact-v3 public and withheld GPU smoke and review are still required before any scored agent launch. The seed is an answer key for optimization, not blind from-spec trials. |
| MHC fused post/pre | Guarded analyst HIP passed 9/9 cases, but graph replay was 20-106x slower than AITER. | No credible HIP parity seed; unscored. |
| Sparse prefill | Analyst HIP passed 5 public and 4 withheld correctness cases. Read-only post-graph verification passed, but two public graph buckets were 9.97x and 11.86x slower. | No credible HIP parity seed; unscored. |
| OPUS persistent A16W16 GEMM | Three original exact-kid cases passed the independent Torch oracle and the [source-equivalent standalone HIP adapter](../tasks/opus_a16w16_persistent/FEASIBILITY.md). Eight full-K public cases passed guarded correctness and two independent [graph-control sessions](../tasks/opus_a16w16_persistent/scored_candidate/SECOND_SESSION.md), with second-session adapter/AITER ratios 0.97739-1.01087. An unchanged-header [production `opus_bmm` one-case control](../tasks/opus_a16w16_persistent/scored_candidate/PRODUCTION_SEED_SMOKE.md) passed independent six-step oracle/guards, exact persistent dispatch, and paired graph timing at 0.983445 candidate/AITER. The [draft task](../tasks/opus_a16w16_persistent/scored_candidate/README.md) freezes eight full-K public cases and a committed off-repo twelve-case withheld matrix. [Batch attempt 002](../tasks/opus_a16w16_persistent/scored_candidate/PRODUCTION_BATCH_INTERRUPTION.md) has only two observed public passes; its later results and exit status became unreachable. Partial-K remains excluded after the AITER-only [padding-dependent defect](../tasks/opus_a16w16_persistent/K_TAIL_REPRO.md) filed as [ROCm/aiter #5954](https://github.com/ROCm/aiter/issues/5954). | Plausible source-preserving route for the eight full-K cases only. Neither the one-case control nor the two observed batch-prefix passes establish all-eight parity. Reconcile the remote batch before further OPUS admission; `scored_eligible=false`, zero OPUS agent trajectories. |
| TopK long-row | Pinned FlyDSL route matched an independent oracle on 20 calls. | Not a HIP pilot without a parity adapter. |

The separate AITER GDR `scale=NaN` validation bypass is tracked in the
[audit receipt](2026-09-29-gdr-scale-nan.md) and upstream issue #5951. It is
an upstream-wrapper finding, **not** an agent-written-kernel mistake. The
[fe2o3 IR control](2026-09-29-fe2o3-wave-ir-gap.md) rejects an explicit
partial-lane collective claim but accepts a false full-wave claim inside a
lane-divergent branch; this is not an end-to-end Rust-source proof. The
[paired `gfx942` source check](../cases/fe2o3/source_wave_participation/README.md)
found both divergent and uniform safe-collective variants typecheck, while
the current Wave64 source-to-KIR route is retired. That is an unsupported
source-verification path, not a demonstrated fe2o3 catch or gfx950 result.

## Scale and next gates

The initial 36-trajectory scored pilot has six eligible captures, all on one
task. The
[gfx950 census](../inventory/results_census.md) contains source-entry and
configuration hints, not 10,000 distinct vetted HIP challenge types. Do not
inflate type count with shape rows, repeated attempts, or AITER-only audits.

Next: run and review exact-v3 GDR public and withheld GPU smoke, then commit
the separately gated admission receipt before any scored agent launch. The
integrated v3 CPU protocol tests pass 33/33, but no v3 GPU or agent run has
occurred. Access to `mi350-2` currently fails at hostname resolution. All
four earlier GDR exploratory captures remain outside the incidence
denominator. Add a final per-run adjudication row before estimating rates:
agent-introduced, AITER baseline, harness/environment, unsupported, or
unresolved, with diff/reproducer hashes, functionality and performance
outcomes, and an explicit cohort-inclusion decision. For megakernels,
complete the staged eight-public/twelve-withheld
production-overlay gate before any scored OPUS attempt. The standalone adapter
is only a source-equivalent control, and the single production bucket is not
an all-case parity result.
Keep FlyDSL-only and binary-only megakernel paths outside source-level HIP
agent-error incidence until a legitimate same-boundary route exists.
