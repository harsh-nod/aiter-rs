#ifndef AITER_RS_MHC_HIP_ABI_H
#define AITER_RS_MHC_HIP_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All pointers are device pointers except stream. Inputs/outputs are contiguous.
 * BF16: layer_input [m,h], residual_in/next_residual [m,4,h],
 * layer_input_out [m,h], norm_weight [h] or NULL.
 * FP32: post_layer_mix [m,4], comb_res_mix/comb_mix [m,4,4],
 * fn [24,4h], hc_scale [3], hc_base [24], post_mix [m,4,1].
 * The launch is asynchronous on stream and returns 0 on success.
 */
size_t aiter_rs_mhc_workspace_bytes(int64_t m, int64_t hidden_size, int use_norm);

int aiter_rs_mhc_post_pre(
    const void* layer_input,
    const void* residual_in,
    const void* post_layer_mix,
    const void* comb_res_mix,
    const void* fn,
    const void* hc_scale,
    const void* hc_base,
    const void* norm_weight,
    void* post_mix,
    void* comb_mix,
    void* layer_input_out,
    void* next_residual,
    void* workspace,
    size_t workspace_bytes,
    int64_t m,
    int64_t hidden_size,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat,
    float norm_eps,
    void* stream);

#ifdef __cplusplus
}
#endif

#endif
