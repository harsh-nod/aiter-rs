# Read-only post-replay control

This 2026-09-29 rerun supersedes the post-graph correctness assertion in
[`GRAPH_FEASIBILITY_RECEIPT.md`](GRAPH_FEASIBILITY_RECEIPT.md). The probe now
uses a direct operator call only for preflight. Its final check reads the
existing output, guards, and inputs **without** launching either operator or
filling the output. A CPU regression test simulates a corrupt graph output
that a new direct call would repair; the read-only inspector rejects it.

The AITER SHA remained `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`,
the analyst HIP source SHA256 remained
`7c9f60721e952f975e5490c0eda37ea4ced3232958b134f462e7890709f9d61a`,
and the candidate binary SHA256 remained
`44894188eaf7134bfaf750c47499652d6fd4d5d2d7d9793fff04c755b7a01b4d`.
The corrected probe source SHA256 is
`b51737e4bc3213d1f4d9a30189881ea450e87b619811bb05fb00f811342b33da`.

The two public buckets again used 32 captured calls, five warmup graph
replays, and 20 alternating pairs, with GPU events around each replay. Both
implementations passed the direct preflight and **read-only post-graph** CPU
oracle, output-guard, and unchanged-input checks in both buckets. Every
comparison had zero mismatched elements at the scorer tolerance.

| Public bucket | AITER median (us/call) | HIP median (us/call) | HIP/AITER | AITER MAD/median | HIP MAD/median | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mixed_sources` | 9.104 | 107.960 | 11.859 | 0.034% | 0.017% | Fail |
| `tile_tail_63_64_65` | 12.195 | 121.565 | 9.969 | 0.087% | 0.020% | Fail |

Both buckets pass the 5% MAD criterion but fail the required at-most-1.05
HIP/AITER ratio. The private raw-result SHA256 commitment is
`6e16c644873cb20b8cedc2c740adadac9f1061d11b68ad72b4514b209f871972`;
raw samples remain outside the repository. GPU-event bracketing may include
host enqueue delay, so this is an unscored feasibility comparison, not an
isolated kernel-only latency claim or evidence about agent bug incidence.
