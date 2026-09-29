# GDR HIP pilot: unscored performance feasibility

**Methodology correction:** the
[2026-09-29 host-gap control](2026-09-29-event-timing-host-gap.md) proves that
the event interval can include host enqueue delay. AITER's Python wrapper
and the candidate's direct HIP ABI are unequal launch paths; the apparent
speed margin below may be dominated by that difference. These measurements
cannot qualify device-kernel performance parity. A graph-replay or profiler
control is required before treating this as a credible HIP speed route.

Two unchanged runs of the [predeclared timing probe](../harness/GDR_FEASIBILITY.md)
compared the existing unscored GDR pilot binary (SHA256
`3b7ed6d207f441c7a337279ff916ca80950eac81c2ef56f963284cb58407f60d`)
with AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39` on the same
MI350X/gfx950 and ROCm image. The public GDR spec SHA256 was
`fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`.
Both runs passed all four public correctness cases before timing, including
state/output oracle checks; neither used withheld cases or became a scored
agent trial.

Each run used five warmups and 20 alternating-order paired GPU-event samples
per bucket. Stateful buffers were independently preallocated and reset to the
same initial state **outside** each event window. The same full-call boundary
was timed for the AITER wrapper and HIP ABI launch. The preregistered screen
required candidate median latency <=1.05x AITER in each bucket and each side's
MAD/median <=5%.

| Run | Public bucket | AITER median us | HIP median us | HIP/AITER | Largest MAD | Qualified |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 01 | valid slots | 67.742 | 34.901 | 0.515 | 3.90% | yes |
| 01 | strided mixed | 66.881 | 34.661 | 0.518 | 4.66% | yes |
| 02 | valid slots | 65.341 | 35.301 | 0.540 | **5.42%** (AITER) | **no** |
| 02 | strided mixed | 65.461 | 35.160 | 0.537 | 4.06% | yes |

Both replays showed a large apparent speed margin for this specific
correctness-first HIP implementation, but only run 01 passed the frozen noise
gate. Run 02 is a rejected measurement, not a parity pass. The raw result
SHA256 commitments are `0fd6c33c9327bf948d5d88a609af17e33d54e8c46d0e5a287d3d9406c955cf92`
and `bc0468a1e5d4ba464b958386f796cd3cb34487e21455da535bb7597867028e90`
in order; raw samples remain off-repository on `mi350-2`.

**Interpretation:** a HIP parity route looks plausible for this operator, but
this event-only evidence cannot establish it. A later
[32-call graph-replay control](2026-09-29-gdr-graph-feasibility.md) found the
same naive HIP candidate about 5.8x **slower** than AITER in both public
buckets, with repeatable noise-qualified samples. This is still an unscored
feasibility pilot, not one of the 36 planned trajectories. Before freezing a
scored GDR task, add hidden correctness and repeated-state timing buckets,
validate output/state guards across timing, and demonstrate a plausible HIP
parity route under a preregistered device-work measurement protocol.
