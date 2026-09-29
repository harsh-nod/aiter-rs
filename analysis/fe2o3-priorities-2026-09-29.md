# fe2o3 priorities from the gfx950 pilot

This is an evidence-ranked *pilot* decision, not a population-frequency
estimate. Six scored no-feedback agent captures cover one MXFP4 Even HIP
task; three additional GDR native-seed attempts are unscored optimization
previews on one other task. No agent megakernel trajectory or cross-workgroup
synchronization failure has yet been observed.

| Candidate fe2o3 check | Evidence | Current disposition |
| --- | --- | --- |
| Authenticate wave collective participation and every requested source lane through lane-varying control flow. | Three of six quant final submissions silently corrupted packed output. Two one-line analyst repairs and one within-run snapshot progression support a source-lane participation mechanism. A one-wave HIP control reproduced wrong values; the [fe2o3 IR gate](../results/2026-09-29-fe2o3-wave-ir-gap.md) rejects an honest partial-active-lane claim but accepts a false full-wave claim inside a divergent branch. | **Highest demonstrated semantic-check priority.** Require a Rust-source-to-IR negative/positive pair, authenticated convergence evidence, and gfx950 lowering/ISA validation before crediting fe2o3 with catching these HIP mistakes. Do not infer cross-wave guarantees from this case. |
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
parity route; source-level fe2o3 negative/positive tests for the quant wave
failure; and explicit hardware/compiler assumptions for any cross-workgroup
or forward-progress theorem.
