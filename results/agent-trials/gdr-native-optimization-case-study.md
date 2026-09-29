# GDR native-seed optimization: exploratory case study

These three [unscored no-feedback
previews](gdr-native-optimization-previews.md) optimized one AITER-derived HIP
kernel, not three kernel types. The [trusted replay
receipt](gdr-native-optimization-replay-20260929.md) establishes tested
correctness and public graph-performance outcomes, but not a scored parity or
agent-error-incidence result. Source-level causes of latency remain hypotheses.
The final source hashes and full private-capture commitments are in the capture
receipt. `kernel.hip` line numbers below refer to each immutable private final
snapshot, identified by replicate and snapshot number; no raw trajectory or
private case is published here.

## Submitted strategies

| Replicate | Captured progression | Final source evidence | Optimization attempted |
| --- | --- | --- | --- |
| `preview01` | Starter snapshot 1, final snapshot 2 | `kernel.hip:56-79,181-276` | Compute gate values on wave lane 0; load Q/K on `v_lane == 0` and V on `k_lane == 0`; distribute values with `readfirstlane`/`ds_bpermute`; restrict Q/K normalization and dot product to one eight-lane group. |
| `preview02` | Starter 1; snapshot 2 used an unrefined reciprocal seed; final 3 added one Newton step | `kernel.hip:22-35,94-98,139-145,248-250` | Replace four 256-thread workgroups per head with one 1024-thread workgroup; zero invalid outputs in 16-byte BF16 vectors; replace the sigmoid divide with a reciprocal seed and refinement. |
| `preview03` | Starter 1; snapshot 2 shortened scratch lifetimes and delayed state loads; final 3 added gate broadcast | `kernel.hip:74-85,160-172,183-190,226-263` | Keep the original workgroup shape, remove long-lived packed Q/K and recurrent-float arrays, recompute recurrence from packed state during stores, and compute gate values once on lane 0 per wave. |

The trajectories also describe compile-only alternatives that were **not**
submitted: preview 1 tried an eight-wave/two-tile shape in a temporary copy;
preview 2 explored a 64-value tile before choosing 128; preview 3 tried a
launch-bound variant and Q/K broadcasts in temporary copies, then declined
them. Those experiments are not additional candidates or source snapshots.
All three agents reported successful local `hipcc` compilation and inspected
compiler metadata; those reports are not trusted GPU correctness or latency
measurements.

## Measured outcome

The trusted replay accepted each final source and its pinned task freeze. All
three passed **6/6 visible** checks (including the explicit default-stream-0
case) and **4/4 withheld** stateful correctness cases against the independent
oracle and pinned AITER. This is evidence for the tested domain, not a proof
that no other input can fail. Every public graph bucket was noise-qualified
under the 5% relative-MAD cap. Ratios below are candidate/AITER within the
same session; lower is better, and the public screen is `<=1.05` per bucket.

| Candidate | Valid slots | Strided mixed | Batch 16 | Buckets meeting `<=1.05` |
| --- | ---: | ---: | ---: | ---: |
| `preview01` | 1.276437 | 1.270797 | 1.087302 | 0/3 |
| `preview02` | 1.200599 | 1.221799 | 1.098378 | 0/3 |
| `preview03` | 1.002299 | 1.003089 | 1.062332 | 2/3 |
| `preview03` fresh public repeat, same binary | 1.005677 | 1.005309 | 1.072518 | 2/3 |

Previews 1 and 2 have measured regressions in all three public buckets.
Preview 3 is near the AITER baseline on the first two but exceeds the batch-16
screen in two independent sessions: it was 6.23% and 7.25% slower than AITER,
respectively. Its withheld correctness was not rerun in the second public-only
session. The first replay
and repeat used the same candidate binary; all listed samples passed the
noise gate. The earlier pinned-seed smoke used a different host/container
execution setup, so these within-session ratios should not be read as a
cross-session absolute latency comparison. The measured slowdown does not by
itself identify which agent edit caused it.

## Mechanism hypotheses

**Preview 1: wave-lane participation.** The source uses
`lane = threadIdx.x & 63`, `k_lane = lane & 7`, and `v_lane = lane / 8` from
the unchanged starter. Its `ds_bpermute` source-lane indices target lane
`k_lane` for Q/K and lane `v_lane * 8` for V. This mapping appears deliberate:
the selected producer lanes are active on the valid path, and the invalid-slot
return is block-uniform. Still, correctness now depends on the exact gfx950
wave intrinsic and execution-mask semantics at the reconverged broadcasts.
The visible and withheld replay found no mismatch in its tested valid,
strided, invalid-sentinel, or repeated-state cases, so this is not an observed
wave-semantics error. The 8.7-27.6% graph slowdown could reflect the cost of
wave permutations or masked arithmetic offsetting fewer global loads and
transcendental executions; that explanation is not isolated by this replay.

**Preview 2: numerical and launch-boundary changes.** With 16 waves and
`kVBlocks = 1`, `v_idx = warp * 8 + v_lane` covers all 128 V positions once;
there is no obvious missing V tile in the source. The invalid path changes
from scalar BF16 stores to one 16-byte store by lane 0 of each wave. Check
guarded output coverage and alignment, especially on invalid-sentinel cases;
the tested cases passed.
The reciprocal seed plus Newton step is deliberately not the original divide;
BF16 rounding close to a boundary could change `beta`, then state across
repeated steps, although the tested matrix found no mismatch. A 1024-thread
workgroup can also change scheduling/residency. The 9.8-22.2% graph slowdown
is measured; neither the larger workgroup, vector store, nor reciprocal is
identified as its cause by these bundled edits.

**Preview 3: lifetime versus latency hiding.** The final source moves state
loads after Q/K work and recomputes the recurrent term from packed state for
the store. That may reduce live registers but also remove memory-compute
overlap and add arithmetic. The gate result is produced only on lane 0 and
read with `readfirstlane` after reconvergence; as in preview 1, the tested
cases passed but do not prove every possible lane-mask condition. The
batch-16 slowdown repeated, but this replay does not distinguish delayed
state loads/recomputation from gate broadcast or another code-generation
effect. The extra empty workspace directories and a rejected temporary-file
cleanup command are capture/tool provenance, not kernel findings.

## Analyst controls to run

These are proposed **analyst-only** source-preserving controls, not performed
results. Keep each frozen agent snapshot immutable and compile separate copies
with the same gfx950 flags and trusted host/container setup. Require guarded
public and withheld correctness before graph timing, clean GPU PID gates,
noise qualification, and a fresh-session repeat for close calls.

1. For preview 1, retain its lane-0 gate calculation but restore the starter
   Q/K/V load and reduction path in a separate copy. Compare that gate-only
   control with the frozen final source to test the wave-permutation bundle;
   split Q/K and V broadcasts only if that contrast is informative.
2. For preview 2, make a geometry-only starter variant (`kVBlocks=1`,
   `kWarps=16`) with the original divide and scalar invalid-zero stores.
   Separately restore the original divide in a copy of the frozen final source
   to test the reciprocal contribution without changing launch geometry.
3. For preview 3, replay its captured snapshot 2 (scratch-lifetime change
   only) against final snapshot 3 (scratch plus gate broadcast) on batch 16.
   If the gap persists, use one gate-only copy of the starter to separate the
   scratch rewrite from the gate change.

None of these controls becomes another agent trial or an incidence sample.
They can test mechanisms; they cannot retroactively make the agents' original
performance guesses proven or turn an unscored preview into scored parity.
