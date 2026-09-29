// Deliberately simple analyst baseline. This is not an agent submission.
#include "mhc_hip_abi.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

using bf16 = __hip_bfloat16;

__global__ void post_kernel(const bf16* x, const bf16* residual,
                            const float* post, const float* comb,
                            bf16* next, int m, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * 4 * h) return;
  int col = i % h;
  int out_head = (i / h) % 4;
  int row = i / (4 * h);
  float value = float(x[row * h + col]) * post[row * 4 + out_head];
  for (int in_head = 0; in_head < 4; ++in_head) {
    value += float(residual[(row * 4 + in_head) * h + col]) *
             comb[(row * 4 + in_head) * 4 + out_head];
  }
  next[i] = bf16(value);
}

__global__ void project_kernel(const bf16* next, const float* fn,
                               float* projections, int m, int h,
                               float rms_eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * 24) return;
  int row = i / 24;
  int mix = i % 24;
  int width = 4 * h;
  float dot = 0.0f;
  float square_sum = 0.0f;
  for (int k = 0; k < width; ++k) {
    float value = float(next[row * width + k]);
    dot = fmaf(value, fn[mix * width + k], dot);
    square_sum = fmaf(value, value, square_sum);
  }
  projections[i] = dot * rsqrtf(square_sum / width + rms_eps);
}

__global__ void mix_kernel(const float* projections, const float* scale,
                           const float* base, float* pre, float* post,
                           float* comb, int m, float pre_eps,
                           float sinkhorn_eps, float post_multiplier,
                           int repeat) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  float logits[16];
  for (int k = 0; k < 4; ++k) {
    float a = projections[row * 24 + k] * scale[0] + base[k];
    float b = projections[row * 24 + 4 + k] * scale[1] + base[4 + k];
    pre[row * 4 + k] = 1.0f / (1.0f + expf(-a)) + pre_eps;
    post[row * 4 + k] = post_multiplier / (1.0f + expf(-b));
  }
  for (int k = 0; k < 16; ++k) {
    logits[k] = projections[row * 24 + 8 + k] * scale[2] + base[8 + k];
  }
  for (int a = 0; a < 4; ++a) {
    float mx = fmaxf(fmaxf(logits[a * 4], logits[a * 4 + 1]),
                     fmaxf(logits[a * 4 + 2], logits[a * 4 + 3]));
    float total = 0.0f;
    for (int b = 0; b < 4; ++b) {
      logits[a * 4 + b] = expf(logits[a * 4 + b] - mx);
      total += logits[a * 4 + b];
    }
    for (int b = 0; b < 4; ++b) {
      logits[a * 4 + b] = logits[a * 4 + b] / total + sinkhorn_eps;
    }
  }
  for (int iteration = 0; iteration < repeat; ++iteration) {
    if (iteration > 0) {
      for (int a = 0; a < 4; ++a) {
        float total = 0.0f;
        for (int b = 0; b < 4; ++b) total += logits[a * 4 + b];
        for (int b = 0; b < 4; ++b) logits[a * 4 + b] /= total + sinkhorn_eps;
      }
    }
    for (int b = 0; b < 4; ++b) {
      float total = 0.0f;
      for (int a = 0; a < 4; ++a) total += logits[a * 4 + b];
      for (int a = 0; a < 4; ++a) logits[a * 4 + b] /= total + sinkhorn_eps;
    }
  }
  for (int k = 0; k < 16; ++k) comb[row * 16 + k] = logits[k];
}

__global__ void layer_kernel(const bf16* next, const float* pre, bf16* output,
                             float* layer_fp32, int m, int h, bool norm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * h) return;
  int row = i / h;
  int col = i % h;
  float value = 0.0f;
  for (int head = 0; head < 4; ++head) {
    value += float(next[(row * 4 + head) * h + col]) * pre[row * 4 + head];
  }
  if (norm) layer_fp32[i] = value;
  else output[i] = bf16(value);
}

__global__ void norm_kernel(const float* layer_fp32, const bf16* weight,
                            bf16* output, int m, int h, float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  float sum = 0.0f;
  for (int k = 0; k < h; ++k) {
    float x = layer_fp32[row * h + k];
    sum = fmaf(x, x, sum);
  }
  float inverse = rsqrtf(sum / h + eps);
  for (int k = 0; k < h; ++k) {
    output[row * h + k] = bf16(layer_fp32[row * h + k] * inverse * float(weight[k]));
  }
}

extern "C" size_t aiter_rs_mhc_workspace_bytes(int64_t m, int64_t h, int norm) {
  if (m <= 0 || h <= 0 || m > 4096 || h > 7168) return 0;
  return size_t(m) * (28 + (norm ? h : 0)) * sizeof(float);
}

extern "C" int aiter_rs_mhc_post_pre(
    const void* layer_input, const void* residual_in,
    const void* post_layer_mix, const void* comb_res_mix,
    const void* fn, const void* hc_scale, const void* hc_base,
    const void* norm_weight, void* post_mix, void* comb_mix,
    void* layer_input_out, void* next_residual, void* workspace,
    size_t workspace_bytes, int64_t m, int64_t hidden_size,
    float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, float norm_eps,
    void* stream) {
  bool norm = norm_weight != nullptr;
  size_t needed = aiter_rs_mhc_workspace_bytes(m, hidden_size, norm);
  if (!needed || workspace_bytes < needed || !workspace ||
      !layer_input || !residual_in || !post_layer_mix || !comb_res_mix ||
      !fn || !hc_scale || !hc_base || !post_mix || !comb_mix ||
      !layer_input_out || !next_residual || sinkhorn_repeat < 1) return 1;
  int rows = int(m);
  int h = int(hidden_size);
  auto* projections = static_cast<float*>(workspace);
  auto* pre = projections + rows * 24;
  auto* layer_fp32 = pre + rows * 4;
  auto hip_stream = reinterpret_cast<hipStream_t>(stream);
  post_kernel<<<(rows * 4 * h + 255) / 256, 256, 0, hip_stream>>>(
      static_cast<const bf16*>(layer_input), static_cast<const bf16*>(residual_in),
      static_cast<const float*>(post_layer_mix), static_cast<const float*>(comb_res_mix),
      static_cast<bf16*>(next_residual), rows, h);
  project_kernel<<<(rows * 24 + 255) / 256, 256, 0, hip_stream>>>(
      static_cast<const bf16*>(next_residual), static_cast<const float*>(fn),
      projections, rows, h, rms_eps);
  mix_kernel<<<(rows + 255) / 256, 256, 0, hip_stream>>>(
      projections, static_cast<const float*>(hc_scale),
      static_cast<const float*>(hc_base), pre, static_cast<float*>(post_mix),
      static_cast<float*>(comb_mix), rows, hc_pre_eps, hc_sinkhorn_eps,
      hc_post_mult_value, sinkhorn_repeat);
  layer_kernel<<<(rows * h + 255) / 256, 256, 0, hip_stream>>>(
      static_cast<const bf16*>(next_residual), pre,
      static_cast<bf16*>(layer_input_out), layer_fp32, rows, h, norm);
  if (norm) {
    norm_kernel<<<(rows + 255) / 256, 256, 0, hip_stream>>>(
        layer_fp32, static_cast<const bf16*>(norm_weight),
        static_cast<bf16*>(layer_input_out), rows, h, norm_eps);
  }
  return hipGetLastError() == hipSuccess ? 0 : 2;
}
