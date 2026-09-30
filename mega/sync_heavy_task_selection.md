# Next single-GPU synchronization task: OPUS FP8 MLA decode stage 1

**Source-only selection, not an admitted task.** Pinned AITER:
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. No GPU run, agent capture,
performance comparison, or bug incidence is claimed here.

## Selection and boundary

Select the merged-buffer FP8 [`opus_mla_decode_fp8_fwd` native entry](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus_mla_decode_fwd.cu#L292),
not the allocating `mla_decode_fwd` Python wrapper. It launches one
source-visible gfx950 HIP kernel with preallocated Q/KV, metadata, partials,
final output, and optional final LSE. The Python route is **opt-in** via
`AITER_MLA_USE_OPUS=1`, gfx950, page size 1, merged FP8 Q/KV, and scalar
descales ([dispatch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/mla.py#L904));
do not infer this kernel ran from a generic MLA result. The native dispatch
uses `grid=(work_indptr.size(0)-1,1,1)`, a 512-thread block, and specialized
causal/large-KV variants ([launch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus_mla_decode_fwd.cu#L367)).
Profile the actual symbol/variant and compiler artifact before freezing.

The first admission domain should be page-size-1, contiguous merged FP8
`q[total_q,H,576]` and `kv[total_tokens,576]`, scalar finite positive FP32
`q_scale/kv_scale`, trusted AITER-generated `work_indptr/work_info_set`, and
`H=16, max_seqlen_q=1`. Check actual FP8 encoding and supported metadata
before fixing numerical tolerances. Later `H`, multi-query causal, split-KV,
and large-KV cases are *proposals*, not assumed supported. The native code
selects noncausal specialization whenever `max_seqlen_q==1`, even if `causal`
is true ([dispatch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus_mla_decode_fwd.cu#L411)).

Compare **one stage-1 native call** with one candidate HIP call using the
same pinned metadata and already allocated buffers. The native stage writes
BF16 `out` and optional FP32 `final_lse` for a whole-request work item
(`partial_slot < 0`), or FP32 `logits/attn_lse` for a split work item
(`partial_slot >= 0`) ([writes](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L1220)).
The later `mla_reduce_v1` merge is outside stage-1 latency, but a separate
end-to-end check should verify that all partials merge to the mathematical
attention result. Do not compare the candidate stage alone with an AITER
wrapper including metadata creation, reduction, or allocation.

## Why synchronization is substantive

The [HIP source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L20)
uses eight wave64s per block, four LDS slots, distance-two KV prefetch, one
workgroup `s_barrier` per phase, `vmcnt`/`lgkmcnt` waits, and a parity-dependent
softmax/PV epilogue. The 32-token KV tile and 512-thread block are in
[`mla_decode_traits.h`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_traits.h#L158).
The [prologue](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L985)
primes slots 0-2. In the [phase loop](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L1040),
the VMEM wait and block barrier must publish the prefetched tile before LDS
reads and prevent overwriting a slot until the prior V read finishes. The
[`lgkmcnt` waits](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L1067)
must cover DS reads before their MFMA consumers. The
[one-/even-/odd-tile tail](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L1156)
must drain the last V read/PV and select the right score buffer. All waves in
a CTA must traverse each work item's barriers together. A block may own zero
or multiple work items ([persistent loop](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L1355)).

This is an **intra-CTA** synchronization and state-reuse challenge. The source
comments discuss two waves on one SIMD as a profiling observation; no
one-wave-per-SIMD placement is a correctness premise. Workgroups do not spin
on one another: metadata assigns disjoint work and the next kernel consumes
partials by stream order. It cannot test cross-CTA forward progress or
cross-rank coherence. A proof would still need the actual gfx950 instruction
semantics for barriers, async global-to-LDS loads, wait counts, lane activity,
and MFMA/LDS reads; a correct output corpus is empirical evidence only.

## Proposed adversarial admission

Freeze exact public and private case IDs/seeds **after** source-domain and
runtime-dispatch checks; keep the private manifest outside Git and publish
only its commitment until trials end. Required strata and invariants:

| Stratum | Observation required |
| --- | --- |
| Tile prologue/tail | KV lengths just below, at, and above multiples of 32; 1, 2, 3, 4, and at least 5 tiles so all four LDS slots wrap. Compare worst head/row, not only a tensor mean. |
| Pages and masks | Permuted, repeated, and nonadjacent valid page indices; nonzero sentinels in unused pages; causal multi-query diagonal and noncausal control only after those specializations are admitted. No invalid index may be silently called supported. |
| Work ownership | At least one unsplit direct writer, multiple split partial writers, empty worker, and a worker with multiple queued items. Check exactly the owned `out/final_lse` or `logits/attn_lse` addresses; canaries around all buffers and unchanged inputs/unowned output. |
| Pipeline sensitivity | Scale/QK inputs that move row maxima between tiles and make the softmax rescale branch change; FP8 finite extremes and distinct per-page values so stale LDS, wrong score-buffer parity, or missed V tail are observable. |
| State and liveness | Repeat identical and alternating shapes with poisoned outputs, then independent streams/buffers; synchronize before inspection, bound calls with a watchdog, and classify hangs separately from numerical mismatch. Check graph replay after ordinary launches pass. |

Use an independently implemented FP32 attention oracle on dequantized
operands, including the exact causal convention and per-work-item KV range;
compare AITER and candidate separately to it. Calibrate BF16/FP32 tolerances
on AITER before agents see the task; output guards and input snapshots are
mandatory. Source comments warn that scheduler spills can invalidate the
hand-counted wait budget ([scheduling note](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus/mla_decode_fp8_16mx8_32nx1.hpp#L84)),
so log code-object hash, registers/spills, launch geometry, and exact compiler
flags for each candidate. Tests alone do not prove absence of races.

Only after correctness, dispatch, and guards pass: compare stage-1 device
work using matched preallocated boundaries and paired graph replay with a
declared noise gate. A source-derived, license-preserving HIP seed is the
plausible parity route; a from-scratch attention kernel is not assumed to
reach AITER latency. Require per-bucket non-inferiority (`<=1.05` candidate /
AITER) and report any speedup separately. The header says its remaining idle
time is diffuse, so a 5% improvement objective must be shown measurable and
attainable rather than built into eligibility by assertion. Current status:
`scored_eligible=false`.

## Exclusions and provenance

- MegaMoE V2 uses FlyDSL/MORI symmetric peer memory and `shmem_barrier_all`
  ([source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L35));
  one rank can smoke arithmetic but cannot exercise its cross-rank publisher,
  consumer, release/acquire, or progress protocol. It is not the selected
  one-GPU source-visible HIP task.
- Existing MHC gfx950 has intra-workgroup LDS/barriers but its public fused
  route is a two-kernel operator ([wrapper](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L988));
  split-K CTAs do not synchronize with each other inside a kernel. The study
  has not admitted a parity-capable independent HIP task for it.
- OPUS BF16 MLA decode launches prebuilt code objects rather than the selected
  source-visible header ([native source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mla/opus_mla_decode_fwd.cu#L37)).
  The three-buffer MXFP8 variant adds E8M0 scale/LDS handling and is a later,
  distinct contract.

SHA-256 at the pinned revision: selected HIP header
`ff745af4237ff99f00381c6f2f9314d78436f929e34e09d323413c49ef1c16ff`,
native `.cu` `808226ab338ff0a88cb632ee674675df6b3dc15fa5c2c98092710f8a5664d502`,
traits `c9a6969fcd0f66acb8853037da49007e633cd67c047a9a14921ac91bec56b08f`,
kernel ABI `02f1511becdffc78c59f0dfa4675a44596f442baead1d84d37adf4808d471d0d`,
Python dispatch `6f6a902ab55fd78c8323e848144d181f72b683a4e0f7eaafb7550f529109050a`,
metadata API `f43106377cb946ea3ff9f59ea312987f153f2dd1b556fe778f0bf974de66aa92`.
Excluded-source SHA-256: MegaMoE V2 wrapper
`5bb46766ade32a5cba37367146d8aaecaf342ac4dfb5918dbfefc00409c15c03`,
MHC Python wrapper
`e6d975e3bda664b9169beaba1ed656c7cdc47f6e9d0aa73b44c0ff2fd91e31e0`,
and MHC HIP implementation
`706e681f883a8f80c6564b8e1cf282ae746c410c99b1e74b8cd6b2cb2b77acba`.
