#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t f32_bits(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    return v.u;
}

__device__ __forceinline__ float f32_pow2_normal(int exponent) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(exponent + 127) << 23;
    return v.f;
}

__device__ __forceinline__ bool load_sign_bit(const void* input, int64_t idx) {
    return (reinterpret_cast<const uint16_t*>(input)[idx] & 0x8000u) != 0;
}

__device__ __forceinline__ uint8_t scale_byte_from_amax(float amax, int* exponent_out) {
    const uint32_t bits = f32_bits(amax);
    const uint32_t r_bits = (bits + 0x00200000u) & 0x7F800000u;

    int exponent = -127;
    if (r_bits != 0) {
        exponent = static_cast<int>(r_bits >> 23) - 129;
        exponent = exponent < -127 ? -127 : exponent;
        exponent = exponent > 127 ? 127 : exponent;
    }

    *exponent_out = exponent;
    return static_cast<uint8_t>(exponent + 127);
}

__device__ __forceinline__ uint8_t mxfp4_even_code(float scaled_abs, bool negative) {
    uint8_t mag;

    if (scaled_abs <= 0.25f) {
        mag = 0;
    } else if (scaled_abs < 0.75f) {
        mag = 1;
    } else if (scaled_abs <= 1.25f) {
        mag = 2;
    } else if (scaled_abs < 1.75f) {
        mag = 3;
    } else if (scaled_abs <= 2.5f) {
        mag = 4;
    } else if (scaled_abs < 3.5f) {
        mag = 5;
    } else if (scaled_abs <= 5.0f) {
        mag = 6;
    } else {
        mag = 7;
    }

    return static_cast<uint8_t>(mag | (negative ? 0x8u : 0u));
}

template <typename T>
__device__ __forceinline__ float load_as_f32(const void* input, int64_t idx);

template <>
__device__ __forceinline__ float load_as_f32<__half>(const void* input, int64_t idx) {
    return __half2float(reinterpret_cast<const __half*>(input)[idx]);
}

template <>
__device__ __forceinline__ float load_as_f32<hip_bfloat16>(const void* input, int64_t idx) {
    return static_cast<float>(reinterpret_cast<const hip_bfloat16*>(input)[idx]);
}

template <typename T>
__global__ __launch_bounds__(256) void quant_mxfp4_even_kernel(
    const void* __restrict__ input,
    uint8_t* __restrict__ packed_output,
    uint8_t* __restrict__ scale_output,
    int64_t groups) {
    constexpr int kThreads = 256;
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kThreads / kWaveSize;
    constexpr int kGroupsPerWave = 2;
    constexpr int kGroupsPerBlock = kWavesPerBlock * kGroupsPerWave;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int half = lane >> 5;
    const int lane_in_group = lane & 31;
    const int64_t group = (static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock) +
                          (wave * kGroupsPerWave) + half;
    const bool valid = group < groups;
    const int64_t elem = group * 32 + lane_in_group;

    const float x = valid ? load_as_f32<T>(input, elem) : 0.0f;
    const bool negative = valid ? load_sign_bit(input, elem) : false;
    float amax = fabsf(x);

    amax = fmaxf(amax, __shfl_xor(amax, 16, kWaveSize));
    amax = fmaxf(amax, __shfl_xor(amax, 8, kWaveSize));
    amax = fmaxf(amax, __shfl_xor(amax, 4, kWaveSize));
    amax = fmaxf(amax, __shfl_xor(amax, 2, kWaveSize));
    amax = fmaxf(amax, __shfl_xor(amax, 1, kWaveSize));

    int exponent;
    const uint8_t scale_byte = scale_byte_from_amax(amax, &exponent);
    const float inv_scale = f32_pow2_normal(-exponent);
    const uint8_t code = mxfp4_even_code(fabsf(x) * inv_scale, negative);

    if (valid && lane_in_group == 0) {
        scale_output[group] = scale_byte;
    }

    if (valid && ((lane_in_group & 1) == 0)) {
        const uint8_t hi = __shfl_down(static_cast<unsigned int>(code), 1, kWaveSize);
        packed_output[group * 16 + (lane_in_group >> 1)] =
            static_cast<uint8_t>(code | (hi << 4));
    }
}

}  // namespace

// dtype: 0 = FP16, 1 = BF16. Pointers address distinct device buffers.
// Enqueue on stream and return a hipError_t integer; do not synchronize.
extern "C" int aiter_rs_quant_mxfp4_even(
    const void* input,
    void* packed_output,
    void* scale_output,
    int64_t rows,
    int64_t cols,
    int dtype,
    void* stream) {
    if (dtype != 0 && dtype != 1) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    const int64_t groups = rows * (cols / 32);
    if (groups == 0) {
        return static_cast<int>(hipSuccess);
    }

    constexpr int kThreads = 256;
    constexpr int kGroupsPerBlock = (kThreads / 64) * 2;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((groups + kGroupsPerBlock - 1) / kGroupsPerBlock));
    hipLaunchKernelGGL(
        dtype == 0 ? HIP_KERNEL_NAME(quant_mxfp4_even_kernel<__half>)
                   : HIP_KERNEL_NAME(quant_mxfp4_even_kernel<hip_bfloat16>),
        grid,
        block,
        0,
        reinterpret_cast<hipStream_t>(stream),
        input,
        reinterpret_cast<uint8_t*>(packed_output),
        reinterpret_cast<uint8_t*>(scale_output),
        groups);
    return static_cast<int>(hipGetLastError());
}
