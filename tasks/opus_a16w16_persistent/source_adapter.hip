// Source-preserving standalone launch adapter for pinned AITER gfx950 OPUS.
// Build against the exact pinned AITER source; do not copy or alter its kernel.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>

#include "gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh"

template <bool HasOob>
using PersistentTraits = opus_gemm_a16w16_persistent_traits_gfx950<
    512,
    opus::seq<256, 256, 64>,
    opus::tuple<bf16_t, bf16_t, bf16_t, fp32_t>,
    opus::seq<8, 8, 4>,
    opus::seq<2, 4, 1>,
    opus::seq<16, 16, 32>,
    HasOob,
    0,
    17>;

extern "C" int aiter_rs_opus_a16w16_validate(
    int m, int n, int k, int kid, void* stream_ptr) {
    // HIP's default stream is a valid null handle. Keep it out of rejection checks.
    (void)stream_ptr;
    if (m < 1 || n < 1 || k < 128 || (kid != 300 && kid != 1300) ||
        n % 16 != 0 || k % 2 != 0)
        return static_cast<int>(hipErrorInvalidValue);
    const int loops = (k + 63) / 64;
    if (loops % 2 != 0 || (kid == 1300 && (m % 256 || n % 256 || k % 64)))
        return static_cast<int>(hipErrorInvalidValue);
    if (int64_t(m) * k > std::numeric_limits<int>::max() ||
        int64_t(n) * k > std::numeric_limits<int>::max() ||
        int64_t(m) * n > std::numeric_limits<int>::max())
        return static_cast<int>(hipErrorInvalidValue);
    return static_cast<int>(hipSuccess);
}

extern "C" int aiter_rs_opus_a16w16_persistent(
    const void* a, const void* b, void* y,
    int m, int n, int k, int kid, void* stream_ptr) {
    if (!a || !b || !y) return static_cast<int>(hipErrorInvalidValue);
    const int valid = aiter_rs_opus_a16w16_validate(m, n, k, kid, stream_ptr);
    if (valid != static_cast<int>(hipSuccess)) return valid;

    const int num_tiles_m = (m + 255) / 256;
    const int num_tiles_n = (n + 255) / 256;
    int split_m = (256 + num_tiles_n - 1) / num_tiles_n;
    if (split_m < 1) split_m = 1;
    while (split_m < num_tiles_m && num_tiles_m % split_m != 0) ++split_m;
    if (split_m > num_tiles_m) split_m = num_tiles_m;
    if (num_tiles_m % split_m != 0) return static_cast<int>(hipErrorInvalidValue);
    const int m_per_wg = num_tiles_m / split_m;
    const int m_grp_per_xcd = (split_m + 7) / 8;

    opus_gemm_persistent_kargs_gfx950 args{};
    args.ptr_a = a;
    args.ptr_b = b;
    args.ptr_c = y;
    args.m = m;
    args.n = n;
    args.k = k;
    args.batch = 1;
    args.stride_a = k;
    args.stride_b = k;
    args.stride_c = n;
    args.stride_a_batch = m * k;
    args.stride_b_batch = n * k;
    args.stride_c_batch = m * n;
    args.m_per_wg = m_per_wg;
    args.num_tiles_n = num_tiles_n;
    args.split_m = split_m;
    args.m_grp_per_xcd = m_grp_per_xcd;

    const dim3 grid(num_tiles_n, m_grp_per_xcd * 8, 1);
    const dim3 block(512);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    if (kid == 300) {
        gemm_a16w16_persistent_kernel<PersistentTraits<true>><<<grid, block, 0, stream>>>(args);
    } else {
        gemm_a16w16_persistent_kernel<PersistentTraits<false>><<<grid, block, 0, stream>>>(args);
    }
    return static_cast<int>(hipGetLastError());
}
