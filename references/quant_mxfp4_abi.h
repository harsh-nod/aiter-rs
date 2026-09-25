#ifndef AITER_RS_QUANT_MXFP4_ABI_H
#define AITER_RS_QUANT_MXFP4_ABI_H

#include <stdint.h>

// The scorer owns all device buffers. dtype: 0 = FP16, 1 = BF16.
// Return the hipError_t integer from a launch on stream; do not synchronize.
extern "C" int aiter_rs_quant_mxfp4_even(
    const void* input,
    void* packed_output,
    void* scale_output,
    int64_t rows,
    int64_t cols,
    int dtype,
    void* stream);

#endif
