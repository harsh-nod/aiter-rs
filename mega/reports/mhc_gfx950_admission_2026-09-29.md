# MHC gfx950 admission, correctness only

Pinned AITER: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. Device:
host-attested AMD Instinct MI350X, PCI `0x75a0`, gfx950, 256 CUs, exactly one
GPU exposed. The container's PyTorch device name was empty; independent host
SKU attestation, ROCm product ID, runtime arch, and CU count were checked.
The public case-spec raw SHA256 was
`bfefc746700ab0a3d4dd0e3d7832f320f6664570b43d0c3100d6febcb1d929e5`.
The private raw result SHA256 was
`7a0cee75b85a592a97ce5a58b1db862a1756e620e62a4b6e298576664fe35ddd`.
The raw result and host report remain off Git; this report omits host paths
and environment identifiers.

**Outcome:** 9/9 supported public cases passed. Every element of all four
outputs (`post_mix`, `comb_mix`, `layer_input_out`, `next_residual`) matched the
independent CPU math oracle at `rtol=atol=0.01`; every output had the expected
shape and dtype; all inputs were bitwise unchanged. This is an AITER baseline
admission, not a HIP candidate result or an agent-error observation.

| Case | M | H | Native path observed | Four outputs |
| --- | ---: | ---: | --- | --- |
| `m1_min` | 1 | 512 | fused GEMM + pre | 0 mismatches |
| `m17_tile_tail` | 17 | 512 | fused GEMM + pre | 0 mismatches |
| `m31_tile_tail` | 31 | 768 | fused GEMM + pre | 0 mismatches |
| `m33_rank2` | 33 | 768 | fused GEMM + pre | 0 mismatches |
| `m65_norm` | 65 | 1280 | fused GEMM + pre RMSNorm | 0 mismatches |
| `m127_wide` | 127 | 1024 | fused GEMM + pre | 0 mismatches |
| `m1023_bound_tail` | 1023 | 512 | fused GEMM + pre | 0 mismatches |
| `m1024_bound` | 1024 | 512 | fused GEMM + pre | 0 mismatches |
| `m1025_nonfused_boundary` | 1025 | 768 | large-M `mhc_post` + `mhc_pre_gemm_sqrsum` + pre | 0 mismatches |

The traced entry points were `mhc_fused_post_pre_gemm_sqrsum` and
`mhc_pre_big_fuse` for ordinary fused cases, replacing the latter with
`mhc_pre_big_fuse_rmsnorm` for the norm case. The boundary control traced
`mhc_fused_post_pre_large_m`, `mhc_post`, `mhc_pre`,
`mhc_pre_gemm_sqrsum`, and `mhc_pre_big_fuse`. No fused GEMM appeared in
that control. The actual chosen `(split_k,tile_m,tile_n,tile_k)` configurations
are retained in the raw result.

## Pretrial exclusions

These were proposed shapes tested before the supported matrix was finalized;
native checks aborted the process, so they are **contract discoveries**, not
scored failures or evidence of AITER numerical bugs:

| Proposed shape | Failing native stage | Native check | Source |
| --- | --- | --- | --- |
| `m=31,h=544`, plain | `mhc_pre_big_fuse` | `hidden_size must be divisible by residual_block` | `csrc/kernels/mhc_kernels.cu:1341,1426` |
| `m=65,h=512`, RMSNorm | `mhc_pre_big_fuse_rmsnorm` | `hidden_size only supports 7168, 5120, 4096, 2560 and 1280` | `csrc/kernels/mhc_kernels.cu:2400-2455,2537` |
| `m=1025,h=512`, fallback | `mhc_post` | `hidden_size must be >= residual_block * 2 stages prefetch` | `csrc/kernels/mhc_kernels.cu:1753-1817,1864` |

The final supported-domain validator rejects all three before launch.
The first two whole-matrix attempts had no completed JSON result because
native `AITER_CHECK` terminated the process. The final supported matrix
ran in one process and produced the 9/9 receipt above.

## Remaining gates

No agent HIP candidate, guarded candidate buffers, candidate/AITER cross-check,
or latency samples exist for this task. `performance.status=not_run` and
`scored_eligible=false`. The preallocated two-stage AITER versus full-call HIP
measurement route and candidate ABI are specified in `mega/README.md` and
`mega/mhc_hip_abi.h`, but they remain to be implemented and validated before
any agent trial or performance-parity claim.
