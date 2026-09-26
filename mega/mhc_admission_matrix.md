# MHC fused post/pre: proposed single-GPU admission matrix

Source pin: AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
Status: **source-only proposal, not an admitted or scored task**. No new GPU
run or performance baseline supports this matrix. The earlier
[tail differential](probe_results.md) covered only `M=17,257` on one input
and found no mismatch against unfused HIP; it is not an independent oracle.

## Exact boundary and first contract

The agent challenge is a HIP implementation of the full
[`mhc_fused_post_pre`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L859)
operator, not merely its GEMM stage. Freeze `force_fused=True`,
`w_preshuffle_bf16=False`, `res_preshuffle=False`, `norm_weight=None`,
`hc_mult=4`, `hidden_size=4096`, BF16 `layer_input/residual_in`, and FP32
`post_layer_mix/comb_res_mix/fn/hc_scale/hc_base`. Use the upstream test's
`rms_eps=hc_pre_eps=hc_sinkhorn_eps=1e-6`, `hc_post_mult_value=2.0`, and
`sinkhorn_repeat=20` as the first numerical contract
([inputs](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L798),
[options](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L838)).
Require all four outputs and their layouts: FP32 `post_mix[M,4,1]`, FP32
`comb_mix[M,4,4]`, BF16 `layer_input_out[M,4096]`, and BF16
`next_residual[M,4,4096]`
([allocation/return](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L992)).
The input tensors are not intended mutation targets; test their bytes before
and after. Require distinct input/output storage for the first contract;
arbitrary aliasing is **not** promised by the source.

On the proposed 256-CU gfx950 stratum, the
[policy](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L366)
and [selector](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L535)
predict `(split_k,tile_m,tile_n,tile_k)` of `(64,16,32,32)` for `M=32`,
`(32,32,32,32)` for `M=256`, and `(8,32,32,32)` for `M=1024`. These are
**source-calculated predictions, not observed runtime dispatch**. The wrapper
allocates padded `[split_k,M,32]` GEMM scratch and `[split_k,M]` squared-sum
scratch, then calls
[`mhc_fused_post_pre_gemm_sqrsum`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L1002)
followed by
[`mhc_pre_big_fuse`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L1037).
The first grid has a split-K dimension
([native launch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3430));
the second kernel reduces those partials
([source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L1059)).
This is a fused **operator** implemented with two launches, not a persistent
single-kernel device megakernel.

## Admission matrix

| Gate | Required cases and observation | Pass condition before freezing a scored task |
| --- | --- | --- |
| Dispatch provenance | On the exact gfx950 SKU/compiler, log `get_cu_num()`, chosen tuple, actual two launches, and loaded code-object SHA for `M=32,256,1024`. | All cases take the proposed fused path and 256-CU policy, or the matrix is revised *before* agent trials. `force_fused=False` at `M=1024` is a different path because the wrapper falls back at `M>=1024` ([branch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L911)). |
| Functionality/oracle | Compare all four outputs for `M=32,256,1024`, multiple fixed seeds, zeros, safe finite high-dynamic-range inputs, and a separate Torch mathematical implementation. Cross-check the upstream [`mhc_post_pre_ref`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L735) and unfused HIP separately. | AITER and candidate must both agree with the independent oracle under **per-output** tolerances calibrated and frozen from baseline variability. Reject nonfinite values and catastrophic per-row errors. Upstream [`checkAllclose`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/test_common.py#L518) allows a 5% mismatch ratio by default; do not adopt that aggregate allowance without explicit justification. |
| Tail/shape boundary | Add `M=17,257,1023,1024` as hidden correctness buckets after confirming supported shapes; report last-row and worst-row errors, not only whole-tensor means. Treat `M>1024`, `M=0`, other hidden sizes, packed BF16, residual shuffle, and RMSNorm as outside this first contract. | No masked-row corruption, out-of-bounds access, or unexpected path change. `M=1024` with `force_fused=True` remains in the two-launch path; [`M>1024` takes the large-M wrapper](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L936) and must not be silently scored here. |
| State/stream ordering | Snapshot inputs; run repeated calls with fresh and poisoned scratch, alternating shapes, and two streams with independently owned buffers. Check outputs only after proper events/synchronization. | No input mutation, stale split partial, cross-call contamination, or accidental scratch sharing. The wrapper's allocations and same-stream launch order are part of the baseline ([stages](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L997)); candidate scratch must have an equivalent lifetime. |
| Intra-CTA synchronization | Vary the split-K/tile-row cases above; use watchdog-bounded repeats and sanitizer/instrumentation where supported. Inspect any agent edits affecting async/TDM waits, LDS slot reuse, or both block barriers. | No hangs or data races, and all outputs meet the oracle. The source's partial wait is valid only while later stages remain outstanding, so the tail fully drains ([pipeline/tail](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3233)); two barriers bracket reuse of residual LDS for cross-wave reduction ([source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3378)). Testing detects failures; it does not prove instruction memory semantics. |
| Same-boundary performance | Compare only oracle-correct candidates against the AITER **pair** of low-level native calls with all four outputs and scratch preallocated on both sides. Include both launches, inter-stage stream dependency, split-K reduction, and any candidate kernel count; exclude allocation on both sides or include it on both. Qualify an equal-overhead HIP-event/graph-replay method before trials. | Pin warmup, clocks, input, bucket, repetitions, noise and default non-inferiority threshold (`candidate median <=1.05 * AITER median` **in every required bucket**) as in [PLAN.md](../PLAN.md). Record p95 if relevant. Do not time only GEMM or compare a low-level candidate against AITER's allocating Python wrapper. Graph capture and low-level adapter equivalence are unverified and must be smoke-tested. |

## What this can and cannot teach fe2o3

The fused GEMM declares four warps per CTA at `hc_mult=4`
([kernel/static assertion](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L2543));
its barriers synchronize those waves inside one workgroup. The source does
**not** require a particular mapping of waves to SIMDs or a one-wave-per-SIMD
rule. Its split-K CTAs do not spin on one another; the second launch consumes
their scratch by stream order. Therefore this challenge can expose wrong
wait-count, LDS publication, divergent-barrier, tail-mask, split-offset, and
scratch-lifetime mistakes. It cannot test cross-CTA forward-progress or
multi-GPU release/acquire assumptions. A formal proof would still need a
model/contract for HIP stream ordering, workgroup barriers, async/TDM wait
semantics, LDS visibility, and relevant instruction behavior; the matrix is
an empirical admission and bug-finding protocol, not such a proof.

Unverified prerequisites: exact runtime CU count and selected code objects,
the independent reference's numerical behavior, support for every proposed
tail input, output-specific tolerances, availability and overhead neutrality
of graph/event timing, AITER baseline stability, and a credible from-spec HIP
implementation that can approach parity. Keep `scored_eligible=false` until
these are resolved and the matrix is frozen before any agent trial. AITER
defects discovered during admission are baseline findings, **not**
agent-authored mistakes; later agent failures need their own provenance.
