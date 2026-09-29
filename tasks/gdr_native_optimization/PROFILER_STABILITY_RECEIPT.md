# GDR native seed: profiler and graph stability control

**Unscored analyst control, not an agent trial.** The native-derived HIP seed
comes from pinned AITER's GDR kernel body with only the launch/ABI adaptation
needed by the candidate interface. This check used public batch-4 and batch-6
timing buckets plus the separate proposed batch-16 fixture. The original
four-public/four-withheld correctness matrix and private manifest were not
changed; no withheld case was used in these profiler or graph sessions.

## Exact dispatch boundary

`rocprofv3 1.1.0` kernel and HIP-runtime CSVs were parsed structurally. Each
of the six independent one-call traces had exactly one target
`(anonymous namespace)::gdr_decode_packed_bf16_kernel(...)` dispatch, uniquely
correlated with `hipLaunchKernel`. The kernel symbol was identical across the
AITER and native-derived HIP paths. Each driver call first passed the
independent CPU state/output oracle, guarded input/state/output checks, and
host PID attestation. The profiler's `Grid_Size_X` is global work-items;
the block counts below divide it by the 256-thread workgroup.

| Public bucket | Path | GDR dispatches | Grid blocks | Global work-items | Workgroup |
| --- | --- | ---: | ---: | ---: | ---: |
| valid slots, batch 4 | AITER and HIP, separately | 1 each | 512 | 131,072 | 256 |
| strided mixed, batch 6 | AITER and HIP, separately | 1 each | 768 | 196,608 | 256 |
| valid slots, batch 16 | AITER and HIP, separately | 1 each | 2,048 | 524,288 | 256 |

This is a dispatch/geometry receipt, not a device-time measurement or proof
that the two compiled instruction streams are identical. Early profiler
attempts failed at the container's read-only profiler scratch path and then
at driver preflight; they produced no valid kernel evidence. The owned
profiler process was stopped and its GPU PID cleared before the six valid
traces. The valid traces used a writable private scratch/output mount.

## Independent graph sessions

Two subsequent fresh-container MI350X/gfx950 sessions, separated by another
workload's GPU window, repeated the unchanged trusted 32-call graph replay.
Both passed all five public correctness cases before timing, 32-step CPU
oracle and guards after replay, host PID/no-foreign-process gates, and the
5% relative-MAD noise gate. Each bucket used 20 alternating paired samples
and the same preallocated operator boundary on both paths. Ratios are
HIP/AITER medians; they are not Python-call GPU-event intervals.

| Session | Public bucket | AITER median us | HIP median us | Ratio | Largest relative MAD |
| --- | --- | ---: | ---: | ---: | ---: |
| B01 | batch 4 | 3.5363 | 3.5313 | 0.99859 | 0.29% |
| B01 | batch 6 | 3.4269 | 3.4413 | 1.00419 | 0.24% |
| B01 | batch 16 | 8.6345 | 8.8689 | 1.02714 | 0.69% |
| B02 | batch 4 | 3.5382 | 3.5438 | 1.00159 | 0.25% |
| B02 | batch 6 | 3.4488 | 3.4257 | 0.99329 | 0.24% |
| B02 | batch 16 | 8.7633 | 8.8651 | 1.01163 | 1.50% |

All six buckets passed the `<=1.05` parity screen. Geometric-mean ratios
were 1.00990 and 1.00214. Together with the two earlier unchanged runs in
[batch-16 admission](LARGE_FIXTURE_ADMISSION.md), the four run ratios span
0.99397-1.00159 (batch 4), 0.99329-1.01062 (batch 6), and
1.01163-1.02969 (batch 16). The largest spread is 1.81 percentage points.
A proposed 5% speedup is larger than this observed variation, but cross-day
stability and an actually optimized candidate have not been demonstrated.
The seed has **parity, not improvement**; a no-op must not count as an
optimized kernel. The 5% objective is separate from the required parity
gate. `scored_eligible=false` pending task freeze and sandbox/private-data
admission.

## Provenance commitments

| Artifact | Raw SHA256 |
| --- | --- |
| Pinned AITER revision | `868ccf62a0bcad3aa47f92728340ccb37ed4fb39` |
| Original public spec | `fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145` |
| Separate batch-16 fixture | `10c53ca20475142a87cc515e917194ad07047bc54ef0dd8ca6e665038b023c0b` |
| Native-derived HIP source | `5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519` |
| HIP binary | `8b66dd83364de6bbaed9c254e7d44a3183bc38445413e885ecfa86722870b709` |
| Profiler driver | `052a4de04b300f5539821db3c317273a6546c5a983625b407258f85fcb01b2c4` |
| Structured trace parser | `4ac26368bac49e578bc3e64bb839d4fe968e5062d877417279ae1e2f2c297fa5` |
| Private six-trace receipt | `c078bc086f7fb3d33e0eaaa7b9789dc48ec36d67e415c89fff853a9cc83eff85` |
| Fresh session B01 result | `ebefb3f59bd3368974bcafa053486f730252473e84308d946b76c9cb0b82df20` |
| Fresh session B02 result | `5bec676552e4a1449374ec1c2e15a8136edc61173dcdb03dad4ba218631d57eb` |

The private four-case manifest commitment remains
`4bb61ab49dab61bff62e124cc5554ec92af3a846f41341f77229f58e38e5c9d0`;
it was not opened for these sessions. The six-trace receipt commits each raw
kernel CSV, HIP-runtime CSV, and correctness-driver receipt hash. Raw traces,
samples, and host-identifying manifests remain off-repository. The profiler
image digest was
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
