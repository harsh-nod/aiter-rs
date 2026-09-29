# MHC gfx950 analyst feasibility, unscored

This is an analyst-authored HIP baseline and trusted-adapter control, **not an
agent submission or error-incidence trial**. It uses the same public nine-case
matrix as the pinned AITER admission (`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`),
MI350X PCI `0x75a0`, gfx950, 256 CUs, and the `mhc_hip_abi.h` boundary.
The candidate source is `mega/analyst_naive_mhc.hip`, compiled with `hipcc
-O3 -shared -fPIC --offload-arch=gfx950` in the ROCm image used for admission.

| Artifact | SHA256 |
| --- | --- |
| Public case spec | `bfefc746700ab0a3d4dd0e3d7832f320f6664570b43d0c3100d6febcb1d929e5` |
| Analyst HIP source | `01cd24df45ce76e62e048b1b084c22b9e27453205b5a9a66230f076dea73b59c` |
| Compiled analyst library | `846eb6fee98925f38942573baf6140623f77b2de2ffefad2b22d399ce0815730` |
| Private correctness receipt | `6df7fcc5ce9660ef18e4689b98b8ad14a43e4a3e13a18a0be66c1cf38d69f39e` |
| Private full graph receipt | `5c719b8c0aba79d498f1c5737c6143767bc9e863a39fb69ce760f70dcac863c4` |

The scorer uses separate 256-byte-aligned, canary-guarded GPU allocations for
all inputs, four outputs, AITER scratch, and candidate workspace. The native
baseline directly times preallocated `mhc_fused_post_pre_gemm_sqrsum` plus
`mhc_pre_big_fuse`/`_rmsnorm`; the `m=1025` control instead times preallocated
`mhc_post`, `mhc_pre_gemm_sqrsum`, and `mhc_pre_big_fuse`. No Python allocation
is in either timed call boundary. The HIP candidate is called on the supplied
current HIP stream and covers all four outputs. Every case passed the CPU
oracle and AITER cross-check at `rtol=atol=0.01`, zero mismatched elements in
both reset-and-repeat invocations; input bytes and all guards remained intact.
The same outputs and guards passed again after graph-replay timing.

Each graph contains **32 copies of the exact full-call operation** on one
stream; after five warmups, 20 paired AITER/HIP graph replay samples alternate
order. Per-call GPU-event time is graph duration divided by 32. The GPU PID
gate was clear before and after each bucket; baseline relative median absolute
deviation (MAD) was below the frozen 5% threshold in all buckets.

| Public case | AITER median us | Analyst HIP median us | HIP/AITER | Baseline relative MAD |
| --- | ---: | ---: | ---: | ---: |
| `m1_min` | 8.57 | 186.94 | 21.82 | 0.11% |
| `m17_tile_tail` | 9.08 | 292.43 | 32.21 | 0.06% |
| `m31_tile_tail` | 9.43 | 427.77 | 45.35 | 0.23% |
| `m33_rank2` | 9.46 | 424.85 | 44.93 | 0.11% |
| `m65_norm` | 10.35 | 1099.02 | 106.20 | 0.89% |
| `m127_wide` | 9.87 | 558.55 | 56.56 | 0.48% |
| `m1023_bound_tail` | 14.86 | 307.31 | 20.68 | 0.97% |
| `m1024_bound` | 15.11 | 307.39 | 20.34 | 0.96% |
| `m1025_nonfused_boundary` | 21.43 | 449.30 | 20.97 | 0.06% |

An earlier 20-sample event loop around Python calls produced apparent ratios
of 2.7-14.7x with high baseline CV; short GPU events can absorb CPU enqueue
gaps, as separately confirmed by a host-sleep control. Those samples are
**superseded as device-performance evidence** by graph replay, not combined
with it. Graph replay measures the full captured device operation, not a
per-kernel profiler trace. It does not prove that an optimized HIP candidate
cannot reach AITER; this intentionally simple baseline is far from the frozen
`<=1.05` per-bucket target.

`scored_eligible=false`. Before agent trials, the scorer still needs an
isolated untrusted-library execution boundary, a private correctness matrix,
and a credible optimized HIP candidate or other preregistered evidence that
the parity target is reachable. No candidate speed or bug incidence from this
control should be attributed to agents.
