# Source-visible gfx950 HIP persistent/fused task scout

Pinned AITER: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. This is a
source-only inspection, not a dispatch trace, correctness admission, timing
result, or agent trial. Links below are to that immutable revision. A task
would still need a frozen shape/mode matrix, independent oracle, actual
gfx950 dispatch proof, and a same-boundary graph/profiler performance gate.
In particular, a HIP kernel that implements one stage must not be compared
with the latency of a different whole-operator boundary.

## Ranked shortlist

### 1. OPUS A16W16 persistent BF16 GEMM: best near-term optimization seed

- **Entrypoint and launch.** The public
  [`gemm_a16w16_opus`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/opus/gemm_op_a16w16.py#L391)
  accepts an explicit `kernelId`; the exact-kid `opus_bmm` path used by the
  [regression test](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_opus_a16w16_gemm.py#L83)
  is preferable for a frozen task. The Python JIT binding calls
  `opus_gemm_a16w16_launch` from `module_deepgemm_opus`
  ([binding](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/opus/gemm_op_a16w16.py#L28)).
  The gfx950 [instance table](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/opus_gemm_common.py#L733)
  assigns persistent kids `300..315` and their no-OOB `+1000` variants;
  kid 300 is a 512-thread, 256x256x64 BF16 tile. The
  [generated launcher](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/codegen/gen_instances_gfx950.py#L636)
  fixes the padded XCD-swizzled grid and invokes
  `gemm_a16w16_persistent_kernel`. Explicit kid selection is essential:
  the shape-driven API may choose a different tuned or heuristic kernel.
- **Source and contract.** HIP/C++ device implementation is
  [`opus_gemm_pipeline_a16w16_persistent_gfx950.cuh`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L58).
  Freeze BF16 A/B, B physically `[batch,N,K]`, output BF16 or FP32, no bias,
  even K with at least two and an even number of 64-wide K tiles, a specific
  kid, and the same launch/workspace boundary. The test's
  [Torch FP32 matmul oracle](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_opus_a16w16_gemm.py#L29)
  and exact-kid benchmark are useful seeds, but tolerance and adversarial
  tails still need independent freezing.
- **Synchronization/error value.** The source documents an actual
  [vmcnt/barrier/load-order race](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L15)
  that silently corrupted a particular tile, plus inter-iteration stores,
  shared-memory reuse, wave-uniform early exit, and XCD grid bijectivity.
  These are directly relevant to fe2o3's wave/workgroup reasoning.
- **Parity judgment.** **Plausible for source-preserving HIP optimization,
  unproven.** A standalone HIP shared-library extraction has to carry OPUS
  traits, kargs, codegen choices, and architecture-specific intrinsics;
  a fresh scalar implementation is not a credible parity starting point.

### 2. OPUS FP8 MLA decode stage 1: richest single-GPU synchronization target

- **Entrypoint and launch.** `aiter.mla.mla_decode_fwd` selects
  [`aiter.opus_mla_decode_fp8_fwd`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/mla.py#L907)
  only with `AITER_MLA_USE_OPUS=1`, gfx950, page size 1, merged FP8 Q/KV,
  and scalar descales. The [C ABI host entry](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus_mla_decode_fwd.cu#L292)
  launches a source-compiled HIP kernel, one block per metadata worker, and
  selects causal, large-KV, and wave-spans-tokens specializations. It is
  **stage 1 only**: `mla_reduce_v1` combines partials after the decode
  launch. The similarly named BF16 `opus_mla_decode_fwd` instead loads
  [prebuilt code objects](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus_mla_decode_fwd.cu#L9)
  and is excluded here.
- **Source and contract.** The persistent
  [FP8 HIP pipeline](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L1)
  uses four LDS slots and one workgroup barrier per KV phase. Freeze merged
  576-wide Q/KV, page size 1, work partition metadata, causal mode, head
  count, scalar descales, split-output/LSE/final-output semantics, and the
  exact stage-1 or full-op timing boundary. The
  [MLA test](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mla.py#L437)
  has a reference-backed FP8 decode case; it must be run with the opt-in
  route verified. A stage-1-specific oracle should also check partials and
  LSE, not only post-reduce output.
- **Synchronization/error value.** Manual VMEM/LDS waitcnt budgets,
  four-slot producer/consumer reuse, two-wave-per-SIMD scheduling commentary,
  worker loops, and the 4-GiB descriptor gate give unusually sharp proof
  obligations. The source explicitly warns that compiler spill traffic can
  invalidate a hand-counted waitcnt budget.
- **Parity judgment.** **Possible only as source-preserving stage-1 HIP
  optimization; high setup risk.** Its existing HIP source is a credible
  starting point, but reproducing the tuned pipeline from scratch is not.
  The opt-in dispatch and separate reduce must be proved before admission;
  an ordinary full-op comparison would also include metadata and reduce
  costs that a stage-1-only candidate does not pay.

### 3. OPUS A8W4 MoE stage 2 decode: fused routing/atomic case, not a full megakernel

- **Entrypoint and launch.** The
  [`opus_moe_stage2_a8w4_decode_fwd`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/opus/moe_stage2_a8w4.py#L188)
  wrapper accepts an explicit `kernel_id` and sorted route metadata. Its
  [host implementation](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_moe/include/opus_moe_host_impl.cuh#L390)
  validates mode/layout and dispatches through the
  [gfx950 HIP launcher](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_moe/include/gfx950/a8w4/opus_moe_stage2_a8w4_decode_dispatch_gfx950.cuh#L11).
  The [device pipeline](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_moe/include/gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh#L1218)
  is source-visible and has multiple workgroup barriers and either BF16
  atomic accumulation or a route-output path with a separate reduction.
- **Contract/oracle.** Freeze one exact kid, block-M, FP8 activations,
  packed FP4 weights/scales, sorted route IDs/weights/valid count, output
  mode, and initialization/reduction boundary. The
  [two-stage MoE test](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_moe_2stage.py#L310)
  contains quantized weight preparation and a Torch reference for the full
  MoE, but this scout did not find a dedicated independent stage-2 oracle.
  One must be built before admission, including padded/inactive routes and
  repeated atomic-output calls.
- **Parity judgment.** **Plausible for the exact HIP stage-2 boundary, not
  yet for end-to-end fused MoE.** Production fused-MoE configurations may
  pair this HIP stage with FlyDSL stage 1 and extra sorting/reduction work;
  treating this stage as a source-visible replacement for the complete
  megakernel would be misleading. Atomic output has initialization and
  nondeterministic accumulation-order risks; route-output has a separate
  reduction and different observable layout.

### 4. Fused split GDR update: lower-complexity stateful control task

- **Entrypoint and launch.** The public
  [`fused_split_gdr_update`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/fused_split_gdr_update.py#L37)
  JIT-binds [`fused_split_gdr_update.cu`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/fused_split_gdr_update.cu#L75),
  whose host launch is at line 355. This is generic HIP source, not a
  gfx950-exclusive path, so actual gfx950 dispatch still needs checking.
- **Contract/oracle.** Output and indexed FP32 state are both observable;
  the [CPU reference](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_split_gdr_update.py#L93)
  is a useful starting oracle. Test repeated updates, aliasing/index
  assumptions, L2-norm mode, and unmodified state slots.
- **Parity judgment.** **More plausible to package as standalone HIP, but
  lower megakernel value.** Its `__syncthreads` and 64-lane shuffles expose
  coordination mistakes, though it lacks the deep persistent pipelines
  above. Index collisions and cross-call state mutation need an explicit
  contract. The existing packed-BF16 GDR decode pilot is a distinct kernel.

## Explicit exclusions and next admission move

- PR #4975's binary-only megakernel is **not** a source-visible HIP baseline
  in this pinned tree; no source-level bug attribution or parity promise can
  be made from it. MegaMoEV2 and the gfx950 comm-fused MoE megakernel are
  FlyDSL/MORI paths, not interchangeable with the OPUS HIP stage-2 kernel.
- The BF16 OPUS MLA decode entry above loads `.co` files, whereas its FP8
  sibling is HIP source. Similar names must not be conflated.
- Start with an **exact-kid persistent GEMM source-optimization task** if
  runtime admission confirms the pinned JIT route and graph/profiler parity
  is feasible. In parallel, design a stage-1 MLA FP8 proof matrix around
  barrier participation, waitcnt ordering, LDS slot lifetime, worker
  progress, and descriptor-boundary cases. Do not launch scored trajectories
  until these gates pass; this scout adds zero eligible task types/trials.
