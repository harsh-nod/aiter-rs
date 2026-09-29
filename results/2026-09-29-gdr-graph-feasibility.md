# GDR graph-replay feasibility control

The same unscored, correctness-first HIP GDR pilot binary used in the earlier
[event-only probe](2026-09-29-gdr-performance-feasibility.md) was compared
with pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39` on
`mi350-2` (MI350X/gfx950, PCI `0x75a0`). The binary SHA256 was
`3b7ed6d207f441c7a337279ff916ca80950eac81c2ef56f963284cb58407f60d`;
the public spec SHA256 was
`fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`.
This is an analyst feasibility control, **not an agent trial**.

The [probe](../harness/gdr_perf_probe.py) captures 32 consecutive stateful
calls for each implementation, times graph replay with GPU events, and divides
elapsed time by 32. It resets each implementation's state and clears output
outside every measured interval. After each bucket, the final state/output
and all guards are checked against 32 CPU-oracle steps. All four public GDR
correctness cases passed before timing in both independent runs; the two
timing buckets passed their post-replay oracle/guard checks. Each run used five
direct-call priming passes and five graph-replay warmups, followed by 20
alternating paired samples and the preregistered 5% relative
median-absolute-deviation (MAD) gate. No withheld inputs were used.

| Run | Bucket | AITER median us | HIP median us | HIP/AITER | Largest MAD | Noise qualified |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 01 | valid slots | 3.538 | 20.370 | 5.758 | 0.18% | yes |
| 01 | strided mixed | 3.451 | 20.182 | 5.848 | 0.16% | yes |
| 02 | valid slots | 3.530 | 20.427 | 5.787 | 0.14% | yes |
| 02 | strided mixed | 3.462 | 20.160 | 5.823 | 0.14% | yes |

The raw result SHA256 commitments are
`fcce3b7d564f3e6185c70e92b90358ae8c3e5a7d2b1590ca058662ee349e4930`
and `39b64338d1a6e6d9e87a9ebffa721dc6c628fa9e9d4a3b61807d4ae774cc5ae1`.
Raw samples remain off-repository on `mi350-2`. Both runs failed the frozen
`<=1.05` per-bucket feasibility threshold by a wide margin. Graph replay
amortizes unequal host enqueue gaps but is not a per-kernel profiler trace;
these ratios describe the full captured device operation.

The earlier event-only probe reported an apparent HIP/AITER ratio near 0.52.
The reversal shows why those short-call event intervals were not valid
device-performance evidence. It does **not** show that an optimized HIP GDR
implementation cannot match AITER. This particular naive pilot remains a
correctness control only; GDR is not yet scored-eligible for agent performance
incidence.
