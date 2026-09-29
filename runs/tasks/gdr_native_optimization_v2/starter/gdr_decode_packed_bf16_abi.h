#ifndef AITER_RS_GDR_DECODE_PACKED_BF16_ABI_H
#define AITER_RS_GDR_DECODE_PACKED_BF16_ABI_H

#include <stdint.h>

// Element strides, not byte strides. All pointers reference GPU memory.
// Valid state indices must be unique within one invocation. Negative or
// out-of-range indices produce positive BF16 zero without a state update.
// Launch asynchronously on stream and return a hipError_t integer.
extern "C" int aiter_rs_gdr_decode_packed_bf16(
    const void* mixed_qkv,
    const void* a,
    const void* b,
    const void* dt_bias,
    const void* A_log,
    const int32_t* indices,
    void* state,
    void* out,
    int32_t batch,
    int64_t qkv_row_stride,
    int64_t a_row_stride,
    int64_t b_row_stride,
    int64_t indices_stride,
    int64_t state_slot_stride,
    int32_t state_pool_size,
    float scale,
    void* stream);

#endif
