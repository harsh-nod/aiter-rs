# Next conventional gfx950 task: two-region sparse prefill (BF16, H <= 32)

**Selection status:** source-selected only; **not scored-eligible**. Pinned AITER
revision: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. No runtime dispatch,
baseline latency, oracle agreement, or HIP candidate parity has been measured.
This adds one candidate *contract/algorithm* beyond MXFP4 Even, not a count for
every head count, CSR density, token count, or tuning row.

## Exact AITER path

The public [`pa_sparse_prefill_opus`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/pa_sparse_prefill_opus.py#L102-L183)
uses `get_gfx_runtime()`; gfx950 selects
[`pa_sparse_prefill_gfx950_opus_fwd`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/pa_sparse_prefill_opus.py#L42-L69).
The binding points to
[`opus_mla_v4_prefill_a16w16_gfx950_fwd`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/include/rocm_ops.hpp#L1484-L1495).
For `H <= 32`, its HIP launcher selects **`opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel`**, `KV_TILE=64`, `NUM_WARPS=4`, grid
`(N, ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M), 1)`, on the current HIP
stream. This is a source-compiled gfx950 device template, not the gfx1250
prebuilt path
([launcher](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/py_itfs_cu/mla_v4_prefill_opus_kernels.cu#L110-L139),
[device kernel](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/include/mla_v4_prefill_opus.h#L2104-L2135)).
This existing HIP source makes a same-language performance route credible,
though a new implementation still has to earn parity.

## Candidate contract

Choose **BF16, D=512, H=16 or 32 initially, nonempty N**, with preallocated
`out`; admit intermediate H (such as 17) only after an AITER baseline check.
`q/out` are `[N,H,512]`; prefix `unified_kv` is `[P,512]`, extend `kv` is
`[E,512]`; two int32 CSR `(indptr[N+1], indices[nnz])` pairs select each
token's prefix and extend rows. `attn_sink[H]` is FP32 and `softmax_scale` is
FP32. The launcher requires matching BF16 data/output dtypes, int32 CSR,
`D=512`, contiguous D dimension, contiguous CSR/sink vectors, and equal KV
row strides. CSR indices must be valid row indices; empty rows are allowed
([wrapper contract](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/pa_sparse_prefill_opus.py#L19-L28),
[launcher validation](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/py_itfs_cu/mla_v4_prefill_opus_kernels.cu#L31-L108)).
For the first admission, use fully contiguous row-major tensors and **disjoint
`out` and all inputs**. The kernel argument structure marks input/output
pointers `__restrict__`, so do not score in-place aliasing as supported
([kernel args](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/include/mla_v4_prefill_opus.h#L115-L125)).
The operation mutates only `out`; `N=0` is a launcher no-op, not a meaningful
performance case
([early return](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/py_itfs_cu/mla_v4_prefill_opus_kernels.cu#L80-L84)).

Independent oracle, for token `i`, head `h`, and concatenated selected KV rows
`K_i = concat(unified_kv[prefix_i], kv[extend_i])`:

```text
s_j = softmax_scale * dot(float(q[i,h,:]), float(K_i[j,:]))
w_j = exp(s_j - m) / (sum_k exp(s_k - m) + exp(sink[h] - m))
m = max(max_j s_j, sink[h])
out[i,h,:] = BF16(sum_j w_j * float(K_i[j,:]))
```

The sink contributes to the denominator **but has no value vector**; if both
CSR rows are empty, output is zero. The pinned AITER test has a separate
[PyTorch FP32 reference](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_pa_sparse_prefill.py#L101-L155),
and the device source processes prefix then extend before sink finalization
([device order](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/include/mla_v4_prefill_opus.h#L2142-L2178)).
Use that reference as a cross-check, but maintain an independent local oracle
implementation for scoring. Initial tolerance candidate is the upstream
`atol=rtol=1e-2` plus at most 1% mismatching elements; calibrate on actual
gfx950 AITER-vs-oracle results and reject any all-zero or nonfinite shortcut
([upstream tolerances](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_pa_sparse_prefill.py#L473-L500),
[check](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_pa_sparse_prefill.py#L612-L630)).

## Admission and performance plan

| Role | Proposed cases within this one path |
| --- | --- |
| Visible functional | `N=64,H=16,P=E=256`, seeded BF16 sparse CSR; plus prefix-only, extend-only, and sink-only rows. This mirrors a pinned CI shape ([cases](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_pa_sparse_prefill.py#L647-L676)). |
| Hidden functional | `H=32` and head-tail `H=17` (the latter only after baseline admission); empty/mixed CSR rows, `nnz` around 63/64/65, repeated indices, reversed index order, unequal prefix/extend densities, sink signs/magnitudes, and a second `softmax_scale`. Keep every index in range. Do not infer unsupported aliasing or malformed CSR behavior. |
| Performance buckets | At least `(N,H,P,E)=(64,16,256,256)` and `(128,32,256,256)` with fixed seeded **nonempty** sparse row-length distributions; add a denser long-row bucket only after the first two pass. Record row-length histogram and total `nnz`, since merely changing `P/E` does not fix work. Empty rows are correctness-only. |

Compare **same boundary**: preallocate `out` and all input/CSR buffers; time
the low-level AITER gfx950 binding and the candidate HIP launch on the same
stream, without Python allocation, JIT compilation, reference computation,
or transfer in the timed window. Use the public wrapper separately to confirm
the same branch and contract. A candidate should expose an asynchronous HIP
entry that fills the supplied `out`; exact C ABI, warmup, graph/eager timing,
threshold, and score aggregation belong to harness admission, not this static
note. Capture correctness for each performance bucket before scoring latency.

## Near-duplicate and fallback exclusions

* `H > 32` selects the **other** 8-wave `16mx8_32nx1` algorithm, so do not
  blend it into this mode or use its faster/slower timings as this baseline
  ([branch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/py_itfs_cu/mla_v4_prefill_opus_kernels.cu#L130-L136)).
* FP16 under this branch changes numerical input/output contract. Split
  NoPE-FP8/RoPE-BF16 uses a different public function and device kernel;
  gfx1250 is prebuilt. None is an extra score row for this BF16 task
  ([FP8 entry](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/pa_sparse_prefill_opus.py#L243-L290)).
* TopK decode has a HIP implementation, but the default gfx950 wrapper first
  selects FlyDSL for supported gates and can select another FlyDSL one-block
  route before HIP fallback. A HIP clone of that fallback is **not** a
  same-dispatch performance task without a separately pinned fallback gate
  ([gates](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/topk.py#L490-L516),
  [priority](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/topk.py#L637-L713)).
* CSR seeds, sizes, density, tile-boundary row lengths, and `H=16/32` inside
  this branch are coverage/performance buckets, not additional types. `H=0`,
  invalid indices, cross-device pointers, and aliased input/output are outside
  the proposed admission contract.
