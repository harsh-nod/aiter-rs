// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All BF16 tensors are contiguous row-major with D=512. The caller owns all
// buffers; the implementation must enqueue work on stream and return without
// synchronizing it. Returns hipSuccess (0) or a HIP error code.
hipError_t sparse_prefill_bf16_gfx950(
    const void* q_bf16,                 // [N,H,512]
    const void* unified_kv_bf16,        // [total_pages,512]
    const int32_t* kv_indices_prefix,   // [nnz_prefix]
    const int32_t* kv_indptr_prefix,    // [N+1]
    const void* kv_bf16,                // [total_extend_rows,512]
    const int32_t* kv_indices_extend,   // [nnz_extend]
    const int32_t* kv_indptr_extend,    // [N+1]
    const float* attn_sink,             // [H], denominator-only
    void* out_bf16,                     // [N,H,512], disjoint from all inputs
    int32_t n,
    int32_t h,
    int32_t total_pages,
    int32_t total_extend_rows,
    int32_t nnz_prefix,
    int32_t nnz_extend,
    float softmax_scale,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif
