# fe2o3 priorities from the gfx950 pilot

This is an evidence-ranked *pilot* decision, not a population-frequency
estimate. Six scored no-feedback agent captures cover one MXFP4 Even HIP
task; three additional GDR native-seed attempts are unscored optimization
previews on one other task. No agent megakernel trajectory or cross-workgroup
synchronization failure has yet been observed.

| Candidate fe2o3 check | Evidence | Current disposition |
| --- | --- | --- |
| Authenticate wave collective participation and every requested source lane through lane-varying control flow. | Three of six quant final submissions silently corrupted packed output. Two one-line analyst repairs and one within-run snapshot progression support a source-lane participation mechanism. A one-wave HIP control reproduced wrong values; the [fe2o3 IR gate](../results/2026-09-29-fe2o3-wave-ir-gap.md) rejects an honest partial-active-lane claim but accepts a false full-wave claim inside a divergent branch. A [paired Rust-source control](../cases/fe2o3/source_wave_participation/README.md) found that both divergent and uniform `gfx942` safe-collective fixtures typecheck, while the current source-to-KIR Wave64 route is retired. | **Highest demonstrated semantic-check priority, not a demonstrated fe2o3 catch.** The literal raw shuffle is not available in the safe Rust API, and the analogous safe collective has no current authenticated source-to-KIR disposition. Restore that path, prove source-derived convergence, and validate gfx950 lowering/ISA before crediting fe2o3 with catching the HIP error. Do not infer cross-wave guarantees. |
| Prove logical tensor-row bounds for vector/async tile loads, including physical padding independence. | Pinned AITER OPUS kid 300 accepted partial K; random K=194/254 failed an independent oracle, and changing only row padding changed K=194 output. The [AITER-only repro](aiter_audits/opus/README.md) led to [upstream #5954](https://github.com/ROCm/aiter/issues/5954). | Useful **audit-derived challenge**, not an agent-frequency priority. Distinguish logical row bounds from allocation bounds: a cross-row read can remain inside the allocation yet violate the operator contract. Test only on a fe2o3 kernel/source route that actually supports the corresponding load and layout semantics. |
| Prove cross-workgroup protocols, barriers, memory scope, and forward progress. | The current scored agent cohort contains no such failure. OPUS correctness controls exercise persistent scheduling but are unchanged-source analyst controls, not agent attempts or a progress proof. | High-risk hypothesis for future megakernel tasks, **not ranked by observed agent incidence**. A proof needs an explicit execution/memory model and progress assumptions; no current pilot result validates one-wave-per-SIMD or scheduler fairness assumptions. |

The three [GDR agent previews](../results/agent-trials/gdr-native-optimization-replay-20260929.md)
all passed the tested public and withheld stateful correctness matrix but
regressed on at least one noise-qualified public graph bucket relative to
AITER. Preview03's batch-16 miss repeated in a fresh session. These are
agent-authored *performance* failures in an unscored, no-feedback track, not
evidence that a safety proof should reject the source. A performance gate or
cost model belongs beside, not inside, fe2o3's correctness claims. Do not
count the three previews in the scored incidence denominator or claim a
mechanistic cost cause without an isolated analyst control.

Next evidence needed: independent agent attempts on a source-visible
persistent/synchronizing HIP task with a frozen hidden matrix and same-boundary
parity route; an authenticated source-to-KIR negative/positive result for the
quant wave failure once that route is supported; and explicit
hardware/compiler assumptions for any cross-workgroup or forward-progress
theorem.
