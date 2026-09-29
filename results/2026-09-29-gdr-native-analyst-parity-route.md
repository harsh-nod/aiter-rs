# GDR native-derived HIP parity route (analyst control)

**Scope.** This is an unscored analyst-written control, not an agent kernel or
an agent-error incidence observation. It tests whether the frozen GDR C ABI
can carry a device-work implementation comparable to pinned AITER on one
MI350X (`gfx950`, PCI `0x75a0`). It does not admit GDR as a scored task.

The [analyst HIP source](../references/analyst_native_gdr_decode_packed_bf16.hip)
retains the MIT SPDX/copyright header and the complete anonymous-namespace
device kernel from pinned AITER
[`gdr_decode_packed_bf16.cu`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu).
The device-kernel section was checked byte-for-byte against that revision.
Only the host binding changed: AITER tensor checks/device guard/current-stream
lookup became the frozen pointer/stride/stream C ABI, and the launch returns
`hipGetLastError()` as an integer. The launch geometry remains 256 threads
per block and `batch * 32 * 4` blocks. In particular, the vectorized BF16
loads/stores, LDS swizzles, four-wave mapping, and fast math are unchanged.
HIP permits a zero-valued default-stream handle; an initial adapter version
incorrectly rejected it and failed before launch. The corrected version below
removed that check, with no device-kernel change.

## Correctness and performance

The corrected binary passed the independent CPU state oracle, pinned AITER
comparison, and guarded input/state/output checks on all four public cases.
The trusted correctness-only scorer then passed **8/8 total cases (4 public,
4 withheld)** with the same binary. No withheld identifiers or inputs were
used for performance or published here. The rejected first adapter run is
preserved privately; it is not counted as an agent bug.

Two unchanged graph-replay runs used the predeclared public `valid_slots` and
`strided_mixed` buckets. Each implementation had separately preallocated
state/output, reset outside every timed interval. A graph captured 32
consecutive stateful calls. Five direct-call and five graph warmups preceded
20 alternating-order paired event samples per bucket. After timing, the
independent CPU oracle checked the final output/state after 32 steps, along
with all guards. Both runs passed these checks and the frozen 5% relative
MAD noise gate; the table shows median per-call device-operation intervals.

| Run | Public bucket | AITER us | HIP us | HIP/AITER | Largest MAD |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | valid slots | 3.536 | 3.603 | 1.019 | 1.13% |
| 01 | strided mixed | 3.442 | 3.446 | 1.001 | 0.38% |
| 02 | valid slots | 3.532 | 3.541 | 1.002 | 0.25% |
| 02 | strided mixed | 3.441 | 3.429 | 0.997 | 0.15% |

All four measurements satisfy the exploratory `<=1.05` per-bucket threshold.
This establishes a credible performance route for the candidate ABI: the
previous naive HIP implementation's roughly 5.8x graph-replay deficit is not
an ABI-imposed floor. The result is expected for an unchanged AITER device
kernel and says nothing about whether an independent agent can rediscover it.
Graph replay amortizes host enqueue gaps but is not a per-kernel profiler
trace; a separate profiler control and broader repeated-state/tail timing
matrix are still needed before any scored performance claim.

## Provenance and isolation

- Pinned AITER commit: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Upstream source raw SHA256: `bf5e6dff288af30a56ccddb624287f0e77240d9d1af2aac07bb6313cdad41e2c`.
- Corrected analyst source raw SHA256: `5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519`.
- HIP binary raw SHA256: `8b66dd83364de6bbaed9c254e7d44a3183bc38445413e885ecfa86722870b709`.
- Public task spec raw SHA256: `fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`.
- Container image digest: `sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
- Build command: `/opt/rocm/bin/hipcc -O3 -shared -fPIC --offload-arch=gfx950 analyst_native_gdr_decode_packed_bf16.hip -o libanalyst_native_gdr.so`.
- Private raw result SHA256: first rejected adapter correctness
  `19a0d465a79fedc2d0e6ff64ae30b40d464f6b4baaa646d595f6a7730d49bf62`,
  corrected public correctness
  `4314172d4c75cdcdb41a852b1199b37a873e38547a8ec52dcbfad79d2b09867f`,
  graph runs `771a9c29452e9d7582463db12aacc3f6ce49827a8c47d7f0af7ea1512d00d3ce`
  and `973d1ad0d252bc9cdc8779d913188ee9a302092025ff3b351a4218c300dfe83e`,
  withheld correctness
  `3f8a828119c13b76912d5efceb21272420482db1fc5670088f45b16d8fa7a665`.

The public analyst source is an answer key for this operator. Exclude it and
this report from any future blind agent starter snapshot; agent exposure to
either must be tracked as trial contamination. Raw results and withheld data
remain off-repository. This control is `scored_agent_candidate=false` and
`joint_pass=false` regardless of its exploratory parity result.
