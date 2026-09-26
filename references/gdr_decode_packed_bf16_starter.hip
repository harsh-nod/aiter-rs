// Compilable from-spec HIP starter. It is not a correct kernel or a baseline.
#include "gdr_decode_packed_bf16_abi.h"

#include <hip/hip_runtime.h>

extern "C" int aiter_rs_gdr_decode_packed_bf16(
    const void* mixed_qkv, const void* a, const void* b, const void* dt_bias,
    const void* A_log, const int32_t* indices, void* state, void* out,
    int32_t batch, int64_t qkv_row_stride, int64_t a_row_stride,
    int64_t b_row_stride, int64_t indices_stride, int64_t state_slot_stride,
    int32_t state_pool_size, float scale, void* stream) {
    (void)mixed_qkv;
    (void)a;
    (void)b;
    (void)dt_bias;
    (void)A_log;
    (void)indices;
    (void)state;
    (void)out;
    (void)batch;
    (void)qkv_row_stride;
    (void)a_row_stride;
    (void)b_row_stride;
    (void)indices_stride;
    (void)state_slot_stride;
    (void)state_pool_size;
    (void)scale;
    (void)stream;
    return static_cast<int>(hipErrorNotSupported);
}
