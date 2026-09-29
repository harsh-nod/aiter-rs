# gfx950 persistent A16W16 correctness/dispatch receipt

Status: **correctness/dispatch gate passed; performance and scored-task
admission not run**. Zero agent trials. Pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39` was clean. The probe ran
on `mi350-2` GPU 0, AMD Instinct MI350X (`0x75a0`, `gfx950`) in image
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`
with PyTorch `2.11.0+gitd0c8b1f` and HIP `7.2.53211`. No latency samples
were taken. Raw JSON, JIT cache, and profiler events remain off-repo.

| Public case | `(B,M,N,K)` | kid / path | Persistent M tiles per WG | Torch FP32 oracle, two runs |
| --- | --- | --- | ---: | --- |
| `aligned-nooob` | `(1,8192,4096,256)` | `1300`, no-OOB | 2 | 2/2, zero bad elements |
| `m-tail-oob` | `(1,12287,4096,256)` | `300`, OOB | 3 | 2/2, zero bad elements |
| `n-tail-16-aligned` | `(1,16384,2192,512)` | `300`, OOB | 2 | 2/2, zero bad elements |

For every run, A/B stayed bitwise unchanged, input/output canaries stayed
intact, and outputs were finite. Maximum absolute error was approximately
`0.25` in output value relative to the FP32 result; every element met the
frozen `atol=0.125, rtol=0.02` gate. The exact-kid plans resolved to the
requested IDs with no workspace. GPU profiler symbols contained
`gemm_a16w16_persistent_kernel` for **both** kids, with the expected
`HAS_OOB=false` (`ELb0`) and `HAS_OOB=true` (`ELb1`) specializations.

The first pretrial matrix used `N=2177`, kid 300. It produced nonfinite
output on both runs (28,811 and 28,080 bad elements), while the other two
cases passed. The pinned [persistent tuner](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/opus_gemm_tune.py#L698)
explicitly rejects `N % 16 != 0`, whereas the explicit-kid planner did not
reject this call. Thus this is an **excluded unsupported-domain pretrial**,
not evidence of an agent mistake or a claimed upstream kernel defect. Its
raw result and original spec/probe were preserved privately. The admitted
N-tail case is still partial relative to the 256-column tile but satisfies
the 16-element vector-store alignment.

Provenance hashes:

| Artifact | SHA-256 |
| --- | --- |
| Corrected public `cases.json` / remote spec | `1372ef7f29034b9b3c2f4a43da47ad7618bdbd64df8c05d9903f56cc8c6f794b` |
| Corrected `admit.py` / remote probe | `61cc9805f0ed911cee92a47c9c02aaddb666765886ac9743563ef1b9eb716330` |
| Corrected private raw result | `e612e4a16343bda92e837b9327dfdd3b89ab2f3ed04c0f00391d540f5a477450` |
| Invalid-N pretrial spec | `f0f6230d94fa6bfc9e60963e1e67d27915a96000a12f3d3f314a1f5cde218e69` |
| Invalid-N pretrial probe | `1cc813783c34782797c3bb17492fcde5315d3f4bd2b7d9288a577447c5d59a6a` |
| Invalid-N private raw result | `fad69e57e0822a5d0c28a1edc21419b6d8e961a1720ded088a734c3d64b583d6` |

Before a parity task is eligible, we still need a credible standalone HIP
starting point, independent hidden/adversarial cases, and noise-qualified
same-boundary graph/profiler timings. Neither this receipt nor the existing
source implementation proves those gates.
