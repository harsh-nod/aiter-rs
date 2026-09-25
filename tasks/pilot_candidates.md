# gfx950 pilot candidate shortlist

Pinned AITER revision: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
Links below target that immutable tree. These are six **candidate operator
families**, not six frozen parity-ready agent tasks. None has yet been run or
benchmarked on `mi350-2`; no independent agent trajectory exists. All scored
implementations must be HIP. Before admitting a task, the harness must
capture the actual dispatch path, independent oracle result, AITER latency
distribution, exact GPU SKU and software stack, and a credible HIP parity
route. AITER's tests are useful seeds, not a hidden-test or independent-oracle
substitute by themselves.

| ID | Track and task mode | AITER entry / source | Existing test and useful failure modes | Gate before scoring |
| --- | --- | --- | --- | --- |
| `quant-mxfp4-rounding` | Conventional, from spec | [`quant_mxfp4_hip`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/quant.py#L945), [`quant_mxfp4.cu`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/quant_mxfp4.cu#L32) has gfx950 conversion branches | [`test_quant_mxfp4.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_quant_mxfp4.py#L129) covers reference rounding, layout/shuffle, edge values. Freeze packed-output and scale layout, four modes, tail shapes. | Verify selected mode dispatch and match timed quantization boundary; this is the smallest initial task. |
| `sparse-prefill-csr-sink` | Conventional attention, optimization of HIP source | [`pa_sparse_prefill_opus`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/pa_sparse_prefill_opus.py#L103) explicitly dispatches gfx950 to source-compiled OPUS HIP kernels in [`mla_v4_prefill_opus.h`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/include/mla_v4_prefill_opus.h#L485) | [`test_pa_sparse_prefill.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_pa_sparse_prefill.py#L106) has a Torch reference for two CSR KV regions and sink. Probe empty rows, indices, dtype, and numerical tails. | Source is large; freeze A16W16 D=512 path and shape buckets, then qualify HIP baseline and parity. |
| `topk-decode-gfx950` | Conventional selection, from spec or HIP optimization | [`topk.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/topk.py#L505) gates gfx950 long-row FlyDSL decode by width/rows/k; fallback HIP path also exists. | [`test_topk_per_row.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_topk_per_row.py#L83) compares to Torch top-k; [`test_topk_per_row_stable.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_topk_per_row_stable.py#L29) addresses stable ties. Probe ties, padding, effective length, scratch reuse. | Pin either a long-row FlyDSL dispatch with a HIP parity route, or a HIP-selected regime. Do not mix baselines. |
| `gdr-packed-bf16-state` | Stateful fused decode, HIP optimization | [`gdr_decode_packed_bf16`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L40) explicitly requires gfx950; [`gdr_decode_packed_bf16.cu`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu) is a compact HIP kernel. | [`test_gdr_decode_packed_bf16.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L109) checks reference output/state, invalid sentinels, strided rows, and multi-step BF16 drift. Unique valid indices are a precondition. | Time full output-plus-state update; compare state and untouched slots after repeated calls, not just output. |
| `mhc-post-pre-fused` | Fused post/pre stages, HIP optimization | [`mhc_fused_post_pre`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L859) has a gfx950-specific threshold and calls HIP [`mhc_fused_post_pre_gemm_sqrsum_kernel`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L2543) plus pre-processing. | [`test_mhc.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L810) includes reference and unfused comparisons. Probe M around the gfx950 `1024` dispatch boundary, split-K, BF16 shuffles, and optional RMSNorm. | Freeze small-M fused path separately from large-M two-step path; time same end-to-end operator and outputs. |
| `mega-moe-v2-ep` | Multi-GPU megakernel, HIP from spec or vetted baseline | [`MegaMoEV2`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L31) fuses dispatch, GEMM1, GEMM2, combine via FlyDSL/MORI; gfx950 AOT support is in [`mega_moe.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/aot/flydsl/mega_moe.py#L355). | [`test_mega_moe_v2.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L227) has a numerical reference and adversarial routing/padding cases. Probe fanout, epochs, producer/consumer ordering, skew, ranks, repeated calls. | Requires working multi-GPU MORI setup and a vetted HIP implementation or demonstrated parity route; no source-level attribution from binary-only PR #4975. |

The three rightmost rows are the pilot's fused/stateful/multi-GPU track. The
first three are single-device conventional operators, although sparse
prefill is internally fused attention. These labels are workload strata,
not claims about the number of HIP launches.

## Admission checklist

1. Pin one AITER entry, dispatch branch, required shapes, dtypes, layout,
   numerical tolerance, and all observable outputs or mutations. Record
   source SHA and dependencies.
2. Verify AITER agrees with an independent mathematical reference on every
   required bucket. Mark any AITER/reference disagreement as a baseline
   issue, not an agent error.
3. Establish correctness and timing of the AITER baseline on the **same**
   gfx950 device and boundary that agent HIP code will use. For multi-GPU,
   also pin rank count, topology, collective setup, and stream semantics.
4. Establish a plausible HIP route to each task's frozen non-inferiority
   threshold. Mark infeasible cases excluded or exploratory, never loosen
   the threshold after seeing agent results.
5. Freeze public/hidden tests and task mode; only then count the task as one
   eligible type and start independent agent runs.
