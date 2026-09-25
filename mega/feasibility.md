# Megakernel feasibility on mi350-2

Source revision: AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
Environment available to this study: one exposed MI350X/gfx950 GPU on
`mi350-2` (reported by the runner setup; recheck and log device discovery
before experiments). This is a **source-based task proposal**, not a run
result. No baseline, HIP parity path, or timing noise is yet validated.

## Single-GPU candidate: fused MHC post/pre

Provisional task ID: `gfx950.mhc.post_pre_fused.fp32fn.bf16.v1`.
Use AITER's [`mhc_fused_post_pre`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L859)
with `force_fused=True`, `w_preshuffle_bf16=False`,
`res_preshuffle=False`, and `norm_weight=None`. At `M<=1024` on gfx950 this
enters the fused path, selects a gfx950/256-CU split-K policy
([config](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L366)),
then launches two native HIP kernels: fused post/GEMM/squared-sum
([device kernel](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L2543))
and pre mixing
([device kernel](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L1002)).
This is a fused *operator* with a two-launch implementation, not one giant
device kernel. The `M>1024` branch changes the implementation to `mhc_post`
plus `mhc_pre` under the selected flags, so it is a separate task type.

Proposed first contract, matching the shapes in
[`op_tests/test_mhc.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L798):

- `M in {32, 256, 1024}`, `hidden_size=4096`, `hc_mult=4`, contiguous BF16
  `layer_input[M,4096]` and `residual_in[M,4,4096]`.
- FP32 `post_layer_mix[M,4,1]`, `comb_res_mix[M,4,4]`,
  `fn[24,16384]`, `hc_scale[3]`, and `hc_base[24]`. No shuffled residual,
  packed-BF16 weights, or optional RMSNorm in this first variant.
- Freeze `rms_eps=hc_pre_eps=hc_sinkhorn_eps=1e-6`,
  `hc_post_mult_value=2.0`, `sinkhorn_repeat=20` to the test configuration.
  Return and compare **all four outputs**: FP32 `post_mix[M,4,1]`, FP32
  `comb_mix[M,4,4]`, BF16 `layer_input_out[M,4096]`, and BF16
  `next_residual[M,4,4096]`. There are no intended input mutations.
- Start with `M=256`; use `M=32` and `M=1024` as distinct correctness and
  latency buckets only after checking that each selects the intended path.
  Freeze numerical tolerances independently for each output; do not assume
  BF16 output equality. The AITER test's
  [`mhc_post_pre_ref`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L735)
  is a useful cross-check, not by itself an independent oracle.

The experimental pressure could be a from-spec HIP implementation of the
full operator, or an optimization of a vetted HIP starting implementation
with a frozen speedup target. Do not give agents an already-parity AITER
source tree and count an unchanged submission as a performance improvement.
For fairness, time the same full post/pre boundary on both sides, including
both kernel launches and split-K reduction. Preallocate equivalent outputs
and scratch or include allocation on **both** sides; pin the method before
agent runs. Keep any correct-but-slower attempt in the corpus. Nothing here
establishes the plan's 1.05x parity threshold is achievable.

### Source-derived hazard hypotheses

These are hypotheses for agent mistakes, **not reported AITER defects**:

1. The fused kernel's grid has a split-K dimension and writes split-specific
   `gemm_out_mul` / `gemm_out_sqrsum` scratch
   ([launch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3430));
   the second launch reduces that scratch. Wrong split offsets, padded rows,
   or early scratch reuse can corrupt only some outputs. Test non-multiple
   tile rows and repeated invocations on one and two streams with distinct
   scratch.
2. The GEMM has two LDS staging slots, async/TDM loads, explicit wait counts
   and workgroup barriers. Its own comments state the partial wait proves
   residency only while later stages remain outstanding, then the tail must
   fully drain
   ([pipeline](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3190),
   [tail](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3310)).
   Removing a wait or reusing an LDS slot early could read stale data.
3. A cross-wave reduction reuses LDS scratch and relies on two workgroup
   barriers before warp 0 consumes other warps' deposits
   ([reduction](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3348)).
   An agent may move a conditional return, barrier, or store so participating
   waves disagree or warp 0 reads before publication.
4. The source records an earlier packed-BF16 fragment-bound guard whose
   absence could yield NaNs while `next_residual` stayed correct
   ([static assertion and comment](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L2607)).
   That is a hazard seed for a later packed-BF16 variant, not a bug in this
   first `w_preshuffle_bf16=False` task and not evidence of agent authorship.

This MHC path has **intra-workgroup** synchronization and stream-ordered
inter-kernel dependencies. It does **not** have a cross-CTA spin wait, so
testing it cannot establish whether a GPU guarantees cross-CTA forward
progress or a particular block-residency order.

Admission remains open: verify observed 256-CU gfx950 dispatch, run AITER
and a separately implemented mathematical oracle, resolve all output
tolerances, build a HIP candidate/starter, qualify same-boundary latency and
noise, then freeze tests and threshold. Until then `scored_eligible=false`.

## MegaMoE V2 requires multiple visible GPUs

[`MegaMoEV2`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L31)
uses FlyDSL/MORI SHMEM to fuse dispatch, GEMM1, GEMM2, and combine. Its
constructor accepts `world_size=1`, and a single-rank smoke test may exercise
arithmetic. **It cannot score the cross-rank communication/progress protocol**
on one GPU. The source allocates symmetric/P2P tables and calls
`ms.shmem_barrier_all()`
([workspace](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L263));
with one rank there is no remote publisher/consumer or peer coherence edge.

The minimum meaningful setup for **compact cross-rank** protocol testing is
two peer-accessible gfx950 GPUs exposed to two processes, a working MORI
SHMEM build and symmetric heap, PyTorch distributed process group (the
upstream test initializes Gloo+NCCL and MORI
[here](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L57)),
and a topology/pin that both AITER and HIP candidate use. For V4-Pro's
`experts=384`, two ranks give 192 experts/rank, within the
[source's 256 maximum](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L38).
But the **fixed-slot V4-Pro branch** explicitly requires `world_size=8`
and 48 experts/rank
([config gate](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L402));
two GPUs cannot cover that branch. A full protocol study needs both compact
and fixed-slot cohorts, preferably eight topology-matched gfx950 GPUs.

Concrete source-level progress hazards to study once topology exists:

- Compact Stage1 uses a bounded one-CTA-per-CU producer/consumer cohort;
  fixed-slot oversubscribes and uses arrival tickets to put required
  producers in the first resident cohort
  ([role/grid logic](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L124)).
  Changing role assignment or launch geometry can create a residency
  deadlock. This is a **liveness assumption to test**, not a proof of GPU
  scheduling behavior.
- Tile-ready counters are reset with system-scope stores because publishers
  use system-scope atomics and consumers wait at system scope
  ([reset](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L320)).
  Publisher release, last-contributor acquire, queue write, release, and
  epoch publication form a chain
  ([publication](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L408));
  weaker scope or reordered stores can produce stale payloads or hangs.
- Stage1 waits for `PLAN_READY`, acquires before reading rewritten
  `TILE_EXPECTED`, then waits for tile payload readiness
  ([plan handoff](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L476),
  [tile wait](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L500)).
  Wrong epoch parity, count, or participant number can wait forever on an
  old generation. Vary rank skew, fanout, padding, reused state, streams,
  and repeated invocations; use watchdogs and isolate hangs.

While only one GPU is exposed, perform static protocol review and possibly
one-rank local arithmetic smoke tests, but label them **non-scored for
cross-rank claims**. Do not replace a two-rank experiment with eight CPU
processes targeting the same GPU: it cannot reproduce the peer-memory or
residency topology. No MegaMoE V2 agent error incidence or performance
parity can be reported from the present one-GPU setup.
