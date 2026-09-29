# Analyst graph feasibility receipt

On 2026-09-29, the correct, unscored analyst HIP control was compared with
the pinned AITER gfx950 low-level sparse-prefill call on one gfx950 GPU. The
AITER SHA was `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`, the HIP source
SHA256 was `7c9f60721e952f975e5490c0eda37ea4ced3232958b134f462e7890709f9d61a`,
and the candidate binary SHA256 was
`44894188eaf7134bfaf750c47499652d6fd4d5d2d7d9793fff04c755b7a01b4d`.
The public generator SHA256 was
`66e07e2a38860f6ed4a3e4299d115d8b0cbd82bd36cb810b5cdd11bf7907e8df`.

An initial probe used events *inside* the captured graph. ROCm rejected this
with `External events are disallowed in rocm`; no timings were collected. This
is an invalid infrastructure attempt, not a kernel failure. Its probe source
SHA256 was `249f907b04e21d6e620f5f9af1b3bc45e7af6fdf35149ceb26bc1e48b668f75a`
and private raw-result SHA256 commitment is
`a4982415980c0221c4ea95ce7b257053bae31f812f85228fc07a77c6a8aec390`.

The corrected probe source SHA256 was
`4ec7b5bbe635979c6333397fd9ea1715b399622572c1a27ea9a563000da3de55`.
It used GPU events around each 32-call graph replay, five graph warmups, and
20 alternating AITER/HIP pairs per public bucket. Both implementations passed
the CPU oracle, output guards, and unchanged-input checks before and after
timing, with zero mismatched elements at the scorer tolerance.

| Public bucket | AITER median (us/call) | HIP median (us/call) | HIP/AITER | AITER MAD/median | HIP MAD/median | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mixed_sources` | 9.100 | 108.027 | 11.871 | 0.028% | 0.025% | Fail |
| `tile_tail_63_64_65` | 12.184 | 121.512 | 9.973 | 0.046% | 0.013% | Fail |

Both buckets met the maximum 5% MAD noise criterion but failed the required
HIP/AITER ratio of at most 1.05. The private corrected raw-result SHA256 is
`e66190cca1ae487169c275332d3aa5f939de5f02dfab6f8bc986f2bc592c5d8f`;
raw samples remain outside the repository. GPU-event bracketing can include a
host enqueue gap, so these are feasibility measurements, not an isolated
kernel-only latency claim. This analyst control is not an agent submission and
does not support any claim about agent bug incidence or scored eligibility.
