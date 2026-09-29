// SPDX-License-Identifier: MIT
// Unscored analyst control: simple HIP implementation of the frozen ABI.

#include "../../references/sparse_prefill_abi.h"

#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kHeadDim = 512;
constexpr int kThreads = 256;

using Bf16 = __hip_bfloat16;

__device__ __forceinline__ void consume_row(
    const Bf16* row,
    float q0,
    float q1,
    float softmax_scale,
    int lane,
    float* partial,
    float& maximum,
    float& denominator,
    float& output0,
    float& output1)
{
    const float value0 = __bfloat162float(row[lane]);
    const float value1 = __bfloat162float(row[lane + kThreads]);
    partial[lane] = q0 * value0 + q1 * value1;
    __syncthreads();
    for(int stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if(lane < stride)
            partial[lane] += partial[lane + stride];
        __syncthreads();
    }

    const float score = partial[0] * softmax_scale;
    const float next_max = fmaxf(maximum, score);
    const float old_weight = expf(maximum - next_max);
    const float row_weight = expf(score - next_max);
    denominator = denominator * old_weight + row_weight;
    output0 = output0 * old_weight + row_weight * value0;
    output1 = output1 * old_weight + row_weight * value1;
    maximum = next_max;
    // No thread may overwrite partial[0] before the previous score is read.
    __syncthreads();
}

__global__ __launch_bounds__(kThreads) void sparse_prefill_kernel(
    const Bf16* __restrict__ q,
    const Bf16* __restrict__ unified_kv,
    const int32_t* __restrict__ kv_indices_prefix,
    const int32_t* __restrict__ kv_indptr_prefix,
    const Bf16* __restrict__ kv,
    const int32_t* __restrict__ kv_indices_extend,
    const int32_t* __restrict__ kv_indptr_extend,
    const float* __restrict__ attn_sink,
    Bf16* __restrict__ out,
    int32_t h,
    float softmax_scale)
{
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    const int64_t output_base =
        (static_cast<int64_t>(token) * h + head) * kHeadDim;
    const float q0 = __bfloat162float(q[output_base + lane]);
    const float q1 = __bfloat162float(q[output_base + lane + kThreads]);

    __shared__ float partial[kThreads];
    float maximum = attn_sink[head];
    float denominator = 1.0f;
    float output0 = 0.0f;
    float output1 = 0.0f;

    const int prefix_begin = kv_indptr_prefix[token];
    const int prefix_end = kv_indptr_prefix[token + 1];
    for(int position = prefix_begin; position < prefix_end; ++position)
    {
        const int row_id = kv_indices_prefix[position];
        const Bf16* row = unified_kv + static_cast<int64_t>(row_id) * kHeadDim;
        consume_row(row, q0, q1, softmax_scale, lane, partial,
                    maximum, denominator, output0, output1);
    }

    const int extend_begin = kv_indptr_extend[token];
    const int extend_end = kv_indptr_extend[token + 1];
    for(int position = extend_begin; position < extend_end; ++position)
    {
        const int row_id = kv_indices_extend[position];
        const Bf16* row = kv + static_cast<int64_t>(row_id) * kHeadDim;
        consume_row(row, q0, q1, softmax_scale, lane, partial,
                    maximum, denominator, output0, output1);
    }

    const float inverse = 1.0f / denominator;
    out[output_base + lane] = __float2bfloat16(output0 * inverse);
    out[output_base + lane + kThreads] = __float2bfloat16(output1 * inverse);
}

}  // namespace

extern "C" hipError_t sparse_prefill_bf16_gfx950(
    const void* q_bf16,
    const void* unified_kv_bf16,
    const int32_t* kv_indices_prefix,
    const int32_t* kv_indptr_prefix,
    const void* kv_bf16,
    const int32_t* kv_indices_extend,
    const int32_t* kv_indptr_extend,
    const float* attn_sink,
    void* out_bf16,
    int32_t n,
    int32_t h,
    int32_t total_pages,
    int32_t total_extend_rows,
    int32_t nnz_prefix,
    int32_t nnz_extend,
    float softmax_scale,
    hipStream_t stream)
{
    if(q_bf16 == nullptr || unified_kv_bf16 == nullptr ||
       (kv_indices_prefix == nullptr && nnz_prefix > 0) ||
       kv_indptr_prefix == nullptr ||
       kv_bf16 == nullptr ||
       (kv_indices_extend == nullptr && nnz_extend > 0) ||
       kv_indptr_extend == nullptr || attn_sink == nullptr ||
       out_bf16 == nullptr || n <= 0 || h <= 0 || h > 32 ||
       total_pages <= 0 || total_extend_rows <= 0 ||
       nnz_prefix < 0 || nnz_extend < 0 || !std::isfinite(softmax_scale))
        return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        sparse_prefill_kernel,
        dim3(static_cast<unsigned>(n), static_cast<unsigned>(h)),
        dim3(kThreads),
        0,
        stream,
        reinterpret_cast<const Bf16*>(q_bf16),
        reinterpret_cast<const Bf16*>(unified_kv_bf16),
        kv_indices_prefix,
        kv_indptr_prefix,
        reinterpret_cast<const Bf16*>(kv_bf16),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        reinterpret_cast<Bf16*>(out_bf16),
        h,
        softmax_scale);
    return hipGetLastError();
}
