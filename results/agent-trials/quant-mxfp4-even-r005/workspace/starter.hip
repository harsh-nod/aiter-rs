#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float u32_to_f32(uint32_t x) {
    return __uint_as_float(x);
}

__device__ __forceinline__ uint32_t f32_to_u32(float x) {
    return __float_as_uint(x);
}

__device__ __forceinline__ float half_bits_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;

    if (exp == 0) {
        return u32_to_f32(sign);
    }

    const uint32_t out_exp = exp + (127u - 15u);
    return u32_to_f32(sign | (out_exp << 23) | (mant << 13));
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t b) {
    return u32_to_f32(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ float pow2_int(int exponent) {
    if (exponent <= -127) {
        return u32_to_f32(0x00400000u);
    }
    if (exponent >= 128) {
        return u32_to_f32(0x7f800000u);
    }
    return u32_to_f32(static_cast<uint32_t>(exponent + 127) << 23);
}

__device__ __forceinline__ int scale_exponent_even(float amax) {
    const uint32_t r_bits = (f32_to_u32(amax) + 0x00200000u) & 0x7f800000u;
    if (r_bits == 0) {
        return -127;
    }

    int exponent = static_cast<int>(r_bits >> 23) - 129;
    if (exponent < -127) {
        exponent = -127;
    } else if (exponent > 127) {
        exponent = 127;
    }
    return exponent;
}

__device__ __forceinline__ uint8_t quantize_e2m1_even(float y, uint32_t sign) {
    uint8_t mag;

    if (y <= 0.25f) {
        mag = 0;
    } else if (y < 0.75f) {
        mag = 1;
    } else if (y <= 1.25f) {
        mag = 2;
    } else if (y < 1.75f) {
        mag = 3;
    } else if (y <= 2.5f) {
        mag = 4;
    } else if (y < 3.5f) {
        mag = 5;
    } else if (y <= 5.0f) {
        mag = 6;
    } else {
        mag = 7;
    }

    return static_cast<uint8_t>(mag | (sign ? 0x8u : 0u));
}

__global__ __launch_bounds__(256) void quant_mxfp4_even_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ packed_output,
    uint8_t* __restrict__ scale_output,
    int rows,
    int cols,
    int dtype) {
    constexpr int kGroupSize = 32;
    constexpr int kGroupsPerBlock = 8;

    const int tid = threadIdx.x;
    const int lane = tid & (kGroupSize - 1);
    const int block_group = tid >> 5;
    const int groups_per_row = cols >> 5;
    const int group = static_cast<int>(blockIdx.x) * kGroupsPerBlock + block_group;
    const int total_groups = rows * groups_per_row;
    const bool valid = group < total_groups;

    int input_base = 0;
    int packed_base = 0;

    if (lane == 0 && valid) {
        const int row = group / groups_per_row;
        const int group_col = group - row * groups_per_row;
        input_base = row * cols + group_col * kGroupSize;
        packed_base = row * (cols >> 1) + group_col * (kGroupSize >> 1);
    }

    input_base = __shfl(input_base, 0, kGroupSize);
    packed_base = __shfl(packed_base, 0, kGroupSize);

    uint16_t raw = 0;
    float x = 0.0f;

    if (valid) {
        raw = input[input_base + lane];
        x = (dtype == 0) ? half_bits_to_float(raw) : bf16_bits_to_float(raw);
    }

    float a = __builtin_fabsf(x);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a = fmaxf(a, __shfl_down(a, offset, kGroupSize));
    }

    int exponent = -127;
    float inv_scale = 0.0f;
    if (lane == 0) {
        exponent = scale_exponent_even(a);
        inv_scale = pow2_int(-exponent);
        if (valid) {
            scale_output[group] = static_cast<uint8_t>(exponent + 127);
        }
    }

    inv_scale = __shfl(inv_scale, 0, kGroupSize);
    const float y = __builtin_fabsf(x) * inv_scale;
    const uint8_t nibble = quantize_e2m1_even(y, raw & 0x8000u);
    const int partner = __shfl_down(static_cast<int>(nibble), 1, kGroupSize);

    if (valid && ((lane & 1) == 0)) {
        packed_output[packed_base + (lane >> 1)] = static_cast<uint8_t>(nibble | (partner << 4));
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
    if (input == nullptr || packed_output == nullptr || scale_output == nullptr ||
        rows <= 0 || cols <= 0 || (cols & 31) != 0 || (dtype != 0 && dtype != 1)) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    constexpr int kThreads = 256;
    constexpr int kGroupsPerBlock = 8;
    const int64_t total_groups = rows * (cols >> 5);
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((total_groups + kGroupsPerBlock - 1) / kGroupsPerBlock));

    hipLaunchKernelGGL(
        quant_mxfp4_even_kernel,
        grid,
        block,
        0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const uint16_t*>(input),
        static_cast<uint8_t*>(packed_output),
        static_cast<uint8_t*>(scale_output),
        static_cast<int>(rows),
        static_cast<int>(cols),
        dtype);

    return static_cast<int>(hipGetLastError());
}
