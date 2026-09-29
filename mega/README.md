# gfx950 MHC post/pre pilot

This is a correctness-first admission for AITER `mhc_fused_post_pre`, pinned to
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39` on an MI350X (PCI device
`0x75a0`, gfx950, 256 CUs). It is not yet a scored agent challenge. The public
matrix is `mhc_gfx950_cases.json`; `mhc_oracle.py` independently computes all
four results on CPU, including the BF16 rounding between post and pre. The
admission command checks the actual high-level dispatch, GPU identity, input
immutability, dtype/shape, and every output element at `rtol=atol=0.01`.

The supported pilot domain is contiguous BF16 layer/residual tensors, plain
contiguous FP32 weights/mixes, `hc_mult=4`, `hidden_size>=512` divisible by
256, optional BF16 RMSNorm weight only at hidden sizes 1280, 2560, 4096,
5120, or 7168, and `force_fused=True`. `m<=1024` must use
the gfx950 fused GEMM/square-sum kernel followed by the fused pre kernel;
`m=1025` is a route-boundary control and must use the large-M post/pre
fallback. Dispatch alone is not a correctness result. Other weight packing,
residual shuffle, dtypes, and multi-GPU behavior are outside this pilot.

## Source contract

- `aiter/ops/mhc.py:220` sets the gfx950 fused M upper bound to 1024.
- `aiter/ops/mhc.py:970-1065` selects config, allocates scratch/outputs, and
  invokes the two native fused stages for `m<=1024`.
- `csrc/kernels/mhc_kernels.cu:1340-1385` requires `hidden_size` divisible by
  the pre-kernel residual block (256 on gfx950); lines 3430-3565 define the
  fused GEMM launch and its shape constraints.
- `op_tests/test_mhc.py:365-430,683-760` gives the unfused mathematical
  relation; this pilot's CPU code is separately written and tested.

Before the supported matrix was frozen, a proposed `m=31,h=544` probe reached
the pinned native `mhc_pre_big_fuse` and aborted with
`hidden_size must be divisible by residual_block` at
`csrc/kernels/mhc_kernels.cu:1426`. This is an explicit pretrial domain
exclusion, not an agent-authored bug or a scored result. The case is now
`m=31,h=768`; hidden-size validation enforces the source constraint.
An early `m=65,h=512` RMSNorm case likewise aborted at the pinned
`mhc_pre_big_fuse_rmsnorm` dispatch: `hidden_size only supports 7168, 5120,
4096, 2560 and 1280` (`csrc/kernels/mhc_kernels.cu:2537`). It is another
pretrial exclusion; the frozen norm case uses `h=1280`.
An early `m=1025,h=512` fallback also aborted in native `mhc_post`:
`hidden_size must be >= residual_block * 2 stages prefetch`
(`csrc/kernels/mhc_kernels.cu:1864`). The fallback dispatch chooses a
512-element block at that width; the boundary control now uses `h=768`,
which chooses 256-element blocks. These are three separate contract-discovery
events, not counted failures in the final supported matrix.

## HIP parity route

`mhc_hip_abi.h` defines a candidate boundary that includes the entire
post/pre operation, with all four preallocated outputs, caller-supplied
workspace, and a caller-supplied HIP stream. The candidate may fuse stages
internally but cannot omit a result. For fair AITER timing, preallocate
`gemm_out_pad` `[split_k,m,32]` FP32 and slice to `[split_k,m,24]`,
`gemm_out_sqrsum` `[split_k,m]` FP32, and the four outputs once. Time the
**pair** of native calls `mhc_fused_post_pre_gemm_sqrsum` plus
`mhc_pre_big_fuse` or `_rmsnorm` on the same stream and identical input data;
do not include Python allocation in either measurement. Freeze the gfx950
config from `get_mhc_fused_post_pre_config` before warmup. A candidate launch
is timed across its full call boundary, including all kernels it enqueues.
The `m=1025` fallback is a separate bucket and must compare to the matching
low-level `mhc_post` + `mhc_pre_gemm_sqrsum` + `mhc_pre_big_fuse` sequence,
with its output and scratch buffers also preallocated, not to a fused baseline.

`mhc_score.py` now exercises the C ABI with guarded and 256-byte-aligned
inputs, outputs, and workspace, checks all four results against both the CPU
oracle and pinned AITER, and invokes the candidate twice from reset output and
workspace sentinels. Its `--unscored-analyst` mode is the **only** enabled
mode; `analyst_naive_mhc.hip` is an explicitly analyst-written correctness
baseline, not an agent trial. The candidate library is loaded in the scorer
process, so scored untrusted submissions still require a separate isolated
execution boundary and private-input handling.

Timing is opt-in. Python-call GPU events are retained only as diagnostic
samples because host enqueue gaps can contaminate short operations.
`--graph-repetitions 32` captures each preallocated full-call boundary 32
times and measures graph replays, amortizing host launch overhead; it still
requires quiet-GPU PID checks, post-run output/guard checks, repeated sampling,
and a baseline relative-MAD no greater than 5%. Report raw samples, median,
p95, and each bucket's ratio. The frozen per-bucket target is candidate median
no more than 5% above AITER; an aggregate cannot compensate for a slow tail.
Graph replay or a device-kernel profiler control is required before any
kernel-speed conclusion. No scored eligibility is claimed by this pilot until
a candidate with a credible parity route passes these gates and an isolated
scored harness exists.
