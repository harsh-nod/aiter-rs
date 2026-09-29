# Optimize the gfx950 persistent BF16 GEMM header

Edit only `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh`.
The trusted runner builds your edit in a clean overlay of pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. Keep the production
`aiter.ops.opus.opus_bmm` interface, kid 300/1300 dispatch, and BF16
`[1,M,K] x [1,N,K] -> [1,M,N]` contract. The eight supported full-K cases
are in `public_matrix.json`; `B` is stored by `[N,K]`, not transposed.

Correctness is checked against an independent FP32 matmul oracle, including
tail rows/columns, repeated invocation, reused and alternate output buffers,
input/output canaries, and input immutability. Performance is compared to
pinned AITER on every bucket at the same preallocated `opus_bmm` boundary.
Per-bucket non-inferiority is <=1.05; the separate improvement target is
geometric mean <=0.95. Partial-K, padded row strides, batch>1, and bias are
outside this task. Do not change other files or depend on a build script in
your workspace being run.
