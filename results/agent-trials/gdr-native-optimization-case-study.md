# GDR native-seed optimization: source-only case study

**Draft, pending trusted GPU replay.** These three [unscored no-feedback
previews](gdr-native-optimization-previews.md) optimized one AITER-derived HIP
kernel, not three kernel types. The source observations below are not bug or
performance findings and do not enter the agent-error incidence denominator.
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

## Questions for replay

**Preview 1: wave-lane participation.** The source uses
`lane = threadIdx.x & 63`, `k_lane = lane & 7`, and `v_lane = lane / 8` from
the unchanged starter. Its `ds_bpermute` source-lane indices target lane
`k_lane` for Q/K and lane `v_lane * 8` for V. This mapping appears deliberate:
the selected producer lanes are active on the valid path, and the invalid-slot
return is block-uniform. Still, correctness now depends on the exact gfx950
wave intrinsic and execution-mask semantics at the reconverged broadcasts.
Check valid, strided, invalid-sentinel, and repeated-state cases against the
independent oracle before calling this an error. The performance question is
whether fewer global loads/transcendental executions outweigh many wave
permutations and masked arithmetic in the graph buckets.

**Preview 2: numerical and launch-boundary changes.** With 16 waves and
`kVBlocks = 1`, `v_idx = warp * 8 + v_lane` covers all 128 V positions once;
there is no obvious missing V tile in the source. The invalid path changes
from scalar BF16 stores to one 16-byte store by lane 0 of each wave. Check
guarded output coverage and alignment, especially on invalid-sentinel cases.
The reciprocal seed plus Newton step is deliberately not the original divide;
BF16 rounding close to a boundary could change `beta`, then state across
repeated steps. Check exact output/state bytes on adversarial values. Even if
correct, a 1024-thread workgroup can change scheduling/residency, so compiler
occupancy metadata alone cannot establish a speedup.

**Preview 3: lifetime versus latency hiding.** The final source moves state
loads after Q/K work and recomputes the recurrent term from packed state for
the store. That may reduce live registers but also remove memory-compute
overlap and add arithmetic. The gate result is produced only on lane 0 and
read with `readfirstlane` after reconvergence; as in preview 1, the valid-path
lane participation must be checked on hardware, not assumed from the helper
name. Compare both output and mutated state over repeated updates. The extra
empty workspace directories and a rejected temporary-file cleanup command
are capture/tool provenance, not kernel findings.

## Evidence needed

For each frozen final snapshot, the trusted replay should independently pin
the source/tree hash, compile with the task's fixed gfx950 flags, compare
stateful output and guarded state with the CPU oracle and pinned AITER on
visible plus withheld cases, and record the explicit default-stream-0 check.
Only after correctness passes should the separate 32-call graph protocol
compare all public buckets, including batch 16, with its contention and noise
gates. Record compile failures and timeouts as such. If a case fails, preserve
the raw private result and use a separately labeled analyst minimization to
test the mechanism; do not retroactively edit an agent snapshot or relabel
these previews as scored trials.
