// Correctness-first HIP starting point for the Even/no-shuffle gfx950 task.
// This is a non-agent reference, not evidence of performance parity.

#include "quant_mxfp4_abi.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ inline uint8_t fp4_rne(float value) {
    float a = fabsf(value);
    uint8_t magnitude;
    if (a > 5.0f) magnitude = 7;
    else if (a >= 3.5f) magnitude = 6;
    else if (a > 2.5f) magnitude = 5;
    else if (a >= 1.75f) magnitude = 4;
    else if (a > 1.25f) magnitude = 3;
    else if (a >= 0.75f) magnitude = 2;
    else if (a > 0.25f) magnitude = 1;
    else magnitude = 0;
    return magnitude | ((__float_as_uint(value) >> 31) << 3);
}

template <typename T>
__global__ void quant_mxfp4_even_kernel(
    const T* input, uint8_t* packed, uint8_t* scales, int64_t rows, int64_t cols) {
    int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t groups_per_row = cols / 32;
    if (group >= rows * groups_per_row) return;

    const T* values = input + group * 32;
    float maximum = 0.0f;
    for (int i = 0; i < 32; ++i) maximum = fmaxf(maximum, fabsf(static_cast<float>(values[i])));

    uint32_t rounded_bits = (__float_as_uint(maximum) + 0x00200000u) & 0x7F800000u;
    int exponent = rounded_bits == 0 ? -127 : static_cast<int>((rounded_bits >> 23) & 0xFF) - 129;
    exponent = max(-127, min(127, exponent));
    float scale = exp2f(static_cast<float>(exponent));
    scales[group] = static_cast<uint8_t>(exponent + 127);

    uint8_t* destination = packed + group * 16;
    for (int i = 0; i < 16; ++i) {
        float lo = static_cast<float>(values[2 * i]) / scale;
        float hi = static_cast<float>(values[2 * i + 1]) / scale;
        destination[i] = fp4_rne(lo) | (fp4_rne(hi) << 4);
    }
}

extern "C" int aiter_rs_quant_mxfp4_even(
    const void* input, void* packed_output, void* scale_output,
    int64_t rows, int64_t cols, int dtype, void* stream) {
    if (!input || !packed_output || !scale_output || rows <= 0 || cols <= 0 || cols % 32)
        return static_cast<int>(hipErrorInvalidValue);
    int64_t groups = rows * (cols / 32);
    dim3 block(256);
    dim3 grid((groups + block.x - 1) / block.x);
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
    if (dtype == 0) {
        hipLaunchKernelGGL(quant_mxfp4_even_kernel<__half>, grid, block, 0, hip_stream,
            static_cast<const __half*>(input), static_cast<uint8_t*>(packed_output),
            static_cast<uint8_t*>(scale_output), rows, cols);
    } else if (dtype == 1) {
        hipLaunchKernelGGL(quant_mxfp4_even_kernel<__hip_bfloat16>, grid, block, 0, hip_stream,
            static_cast<const __hip_bfloat16*>(input), static_cast<uint8_t*>(packed_output),
            static_cast<uint8_t*>(scale_output), rows, cols);
    } else {
        return static_cast<int>(hipErrorInvalidValue);
    }
    return static_cast<int>(hipGetLastError());
}
