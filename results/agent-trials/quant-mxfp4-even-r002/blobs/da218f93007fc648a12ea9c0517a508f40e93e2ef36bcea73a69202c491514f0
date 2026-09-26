#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGroupSize = 32;
constexpr int kGroupsPerBlock = 8;
constexpr int kBlockSize = kGroupSize * kGroupsPerBlock;

__device__ __forceinline__ float f32_from_bits(uint32_t bits) {
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint32_t bits_from_f32(float value) {
    return __float_as_uint(value);
}

__device__ __forceinline__ float pow2_i(int exponent) {
    if (exponent >= -126) {
        return f32_from_bits(static_cast<uint32_t>(exponent + 127) << 23);
    }
    return f32_from_bits(0x00400000u);  // 2^-127, the only subnormal scale used here.
}

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;

    if (exp == 0) {
        return f32_from_bits(sign);
    }

    return f32_from_bits(sign | ((exp + 112u) << 23) | (mant << 13));
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return f32_from_bits(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ float input_abs_to_f32(uint16_t raw, bool is_bf16) {
    raw &= 0x7fffu;
    return is_bf16 ? bf16_to_f32(raw) : fp16_to_f32(raw);
}

__device__ __forceinline__ uint8_t mxfp4_even_code(float y) {
    uint8_t code;
    if (y <= 0.25f) {
        code = 0;
    } else if (y < 0.75f) {
        code = 1;
    } else if (y <= 1.25f) {
        code = 2;
    } else if (y < 1.75f) {
        code = 3;
    } else if (y <= 2.5f) {
        code = 4;
    } else if (y < 3.5f) {
        code = 5;
    } else if (y <= 5.0f) {
        code = 6;
    } else {
        code = 7;
    }
    return code;
}

__global__ __launch_bounds__(kBlockSize)
void quant_mxfp4_even_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ packed_output,
    uint8_t* __restrict__ scale_output,
    int64_t total_groups,
    bool is_bf16) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int64_t group = static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + (tid >> 5);

    uint16_t raw = 0;
    float abs_value = 0.0f;
    if (group < total_groups) {
        raw = input[group * kGroupSize + lane];
        abs_value = input_abs_to_f32(raw, is_bf16);
    }

    float amax = abs_value;
    amax = fmaxf(amax, __shfl_xor(amax, 16));
    amax = fmaxf(amax, __shfl_xor(amax, 8));
    amax = fmaxf(amax, __shfl_xor(amax, 4));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    amax = fmaxf(amax, __shfl_xor(amax, 1));

    const uint32_t r_bits = (bits_from_f32(amax) + 0x00200000u) & 0x7F800000u;
    int exponent = -127;
    if (r_bits != 0) {
        exponent = static_cast<int>(r_bits >> 23) - 129;
        if (exponent < -127) exponent = -127;
        if (exponent > 127) exponent = 127;
    }

    if (group < total_groups && lane == 0) {
        scale_output[group] = static_cast<uint8_t>(exponent + 127);
    }

    uint8_t code = 0;
    if (group < total_groups) {
        const uint8_t sign = static_cast<uint8_t>((raw >> 12) & 0x8u);
        const float y = abs_value * pow2_i(-exponent);
        code = static_cast<uint8_t>(sign | mxfp4_even_code(y));
    }

    if (group < total_groups && ((lane & 1) == 0)) {
        const uint8_t lo = code & 0x0fu;
        const uint8_t odd_code = static_cast<uint8_t>(__shfl_down(static_cast<uint32_t>(code), 1));
        const uint8_t hi = static_cast<uint8_t>((odd_code & 0x0fu) << 4);
        packed_output[group * (kGroupSize / 2) + (lane >> 1)] = static_cast<uint8_t>(lo | hi);
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
    if (input == nullptr || packed_output == nullptr || scale_output == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (rows <= 0 || cols <= 0 || (cols % kGroupSize) != 0 || (dtype != 0 && dtype != 1)) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    const int64_t total_groups = rows * (cols / kGroupSize);
    const uint32_t grid_x =
        static_cast<uint32_t>((total_groups + kGroupsPerBlock - 1) / kGroupsPerBlock);
    const hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

    hipLaunchKernelGGL(
        quant_mxfp4_even_kernel,
        dim3(grid_x),
        dim3(kBlockSize),
        0,
        hip_stream,
        static_cast<const uint16_t*>(input),
        static_cast<uint8_t*>(packed_output),
        static_cast<uint8_t*>(scale_output),
        total_groups,
        dtype == 1);

    return static_cast<int>(hipGetLastError());
}
