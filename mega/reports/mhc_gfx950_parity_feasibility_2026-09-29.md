# MHC gfx950 parity feasibility, offline assessment

**Decision:** keep this nine-case MHC task out of the scored agent cohort for
now. The correctness contract and graph-replay comparator are working, but no
independent HIP implementation has demonstrated the frozen `<=1.05` latency
ratio in every bucket. This is an admission decision, not a claim that parity
is impossible. The only candidate measured was an intentionally naive
analyst-written adapter control, not an agent submission.

## What the pinned implementation actually does

For the public plain-FP32-weight gfx950 route, AITER uses **two GPU kernels**
for `m<=1024` (`aiter/ops/mhc.py:989-1061`). The first combines post residual
mixing, BF16 `next_residual` stores, FP32 square sums, and the 24-output
projection GEMM. A workgroup has four wave64 warps, with a head/warp mapping;
the plain-weight branch uses `v_mfma_f32_16x16x4f32`, not the packed-BF16 or
gfx1250 TDM path (`csrc/kernels/mhc_kernels.cu:150-160,2538-2578,3080-3181`).
Two-stage LDS rings overlap global loads with compute; head contributions
cross warps through LDS and barriers before one partial per split is stored
(`:2899-2940,3350-3428`). This does **not** assume one wave per SIMD: it
specifies four waves per workgroup and explicit synchronization, leaving wave
placement to the hardware. The second kernel consumes split-K partials,
computes sigmoid/Sinkhorn mixes, and reduces four BF16 residual heads to the
layer output (`:1180-1385`). RMSNorm uses a specialized second kernel
(`:2370-2537`). The `m=1025` control is instead the three-native-kernel
post, pre-GEMM, and pre-reduction fallback (`aiter/ops/mhc.py:791-858`).

The gfx950 picker fixes `tile_n=32`, chooses split-K and tile M/K by shape,
and explicitly tunes grid fill (`aiter/ops/mhc.py:350-389`). The admission
receipt showed these first-kernel configurations and resulting launch grids:

| Case | `(split_k,tile_m,tile_n,tile_k)` | Grid CTAs `ceil(m/tile_m)*split_k` |
| --- | --- | ---: |
| `m1_min` | `(8,16,32,32)` | 8 |
| `m31_tile_tail` | `(12,16,32,32)` | 24 |
| `m65_norm` | `(20,16,32,32)` | 100 |
| `m1023_bound_tail` | `(8,32,32,32)` | 256 |
| `m1024_bound` | `(4,16,32,64)` | 256 |

These are **dispatch counts, not measured resident occupancy**. On 256 CUs,
the small-M grids cannot fill every CU even with perfect scheduling; the
boundary grids can nominally supply one CTA/CU, subject to register/LDS
limits. The two-stage `x` and residual LDS arrays consume about 10 KiB/CTA
for `(tile_m,tile_k)=(16,32)` and 20 KiB for `(32,32)` or `(16,64)`, before
other resources (`mhc_kernels.cu:2557-2564`). Every split writes disjoint
`next_residual` K segments and FP32 partials. The reduction then reads both;
the scratch alone is `132*split_k*m` bytes (`[split_k,m,32]` plus
`[split_k,m]`, FP32). The implementation shares each loaded residual tile
across 24 projections and computes its square sum once, unlike the analyst
baseline, whose 24 independent projection threads each rescan the residual
and square it, followed by separate mix/layer/norm launches.

## Timing implication and plausible route

In the 32-call graph replay (20 paired samples), AITER's full-operation
medians were 8.57-21.43 us. All nine baseline relative-MAD values were below
1%, comfortably inside the 5% noise gate. The naive analyst candidate took
20.34-106.20x as long; this diagnoses its algorithm and launch structure,
not the difficulty of every HIP implementation. Python-call event timings
were host-gap contaminated and are not used here. See the separate sanitized
feasibility report for hashes, full per-case data, and correctness receipt.

The frozen 5% margin is only about **0.43 us** at `m=1`, **0.52 us** for the
RMSNorm case, and **0.76 us** at `m=1024`. A credible HIP route therefore
needs approximately the same two-stage launch count (three for the fallback),
not the naive four/five kernels: four wave64 head roles, shape-specific
split-K/tile policy, FP32 MFMA fragments, vectorized/plain BF16 residual
stores, double-buffered LDS with correct async waits and workgroup barriers,
and an efficient split reduction/Sinkhorn kernel. A one-kernel design would
need a different way to resolve split-K partials without an inter-CTA barrier;
no such design has been demonstrated in this study. The source spans roughly
1,000 lines for the fused launch/kernel region and several hundred more for
the reduction, with Opus primitives and architecture-specific variants.
Even after narrowing to plain gfx950, this is **multiple specialized kernels
and likely multiple expert-days of implementation and tuning**, not a small
from-scratch HIP exercise. That time estimate is judgment from code scope,
not a measured development duration.

To admit a scored task, first calibrate the C ABI/graph harness with an
analyst-only AITER-equivalent control, then build or seed an independently
written HIP implementation that passes all four-output/guard cases and the
`<=1.05` target in every bucket, with profiler evidence for occupancy,
MFMA use, memory traffic, and barriers. Freeze that seed and the task mode
before agents run; count only agent-authored changes and failures. Copying
the public AITER kernel verbatim may calibrate a harness but would not make a
meaningful agent-from-spec challenge. Until this route is shown feasible,
`scored_eligible=false`; the task remains useful for correctness-contract
research and for designing synchronization/tail-bug probes, not for parity
incidence statistics.
