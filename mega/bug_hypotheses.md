# Bounded gfx950 AITER bug hypotheses

Pinned AITER: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
These are bounded AITER-audit probes, **not evidence of agent-authored
mistakes**. The one-GPU correctness results are in
[probe results](probe_results.md). They may later inform hidden cases for
agent tasks; do not retroactively add them to frozen agent trials. Run them
only from a checkout of the pinned AITER revision. The GDR probe makes one
operator call; each MHC probe makes one fused and two unfused calls. Neither
performs a latency sweep.

## 1. Empty-batch GDR launch

**Hypothesis.** The public
[`gdr_decode_packed_bf16`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L38)
validates the second dimension of `mixed_qkv` and uses its first dimension
as `batch`, but does not reject `batch=0`
([shape checks](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L71)).
The native launcher computes `grid(batch * 32 * 4)` without a zero-work
return, and its native shape checks also lack a positive-batch precondition
([source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L308)).
A zero-grid HIP launch fails in the pinned single-GPU run rather than treating
empty decode as a no-op. This is a host-launch boundary, not a
synchronization proof.

Run [`probe_gdr_empty_batch.py`](probes/probe_gdr_empty_batch.py) with
`PYTHONPATH=/path/to/aiter` in an environment where AITER builds. It creates
BF16 tensors with shapes `[0,6144]`, `[0,32]`, state `[2,32,128,128]`, and
output `[0,1,32,128]`; it checks that state bytes remain unchanged. The
existing upstream tests use only nonempty batches
([test cases](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L179)).

**Classification gate.** The observed launch error is a *candidate* only. Confirm
whether `batch=0` is in the intended operator domain. The wrapper gives no
explicit positive-batch precondition, but documentation may define one
elsewhere. If unsupported, record a contract/validation gap rather than a
kernel defect. If supported, minimize the result to this one invocation,
pin the HIP error and hardware, and report upstream before using it as a
verifier challenge.

## 2. Non-tile-aligned MHC fused post/pre

**Hypothesis.** The gfx950 fused GEMM computes a partial row tile and uses
`m_oob` for bound checks
([kernel](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L2543),
[tail stores](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3408)).
Its split-K grid writes scratch consumed by a second kernel, and intra-CTA
cross-wave reduction reuses LDS behind barriers
([source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/mhc_kernels.cu#L3348)).
The upstream CLI's default `M` sweep is `1,32,64,128,256,512,1024,...`
([test arguments](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L1115));
it does not exercise `M=17` or `M=257`. A tail mask, split scratch, or
cross-wave publication error could affect only the final row block and be
hidden by an aggregate tolerance.

Run [`probe_mhc_tail.py`](probes/probe_mhc_tail.py) once with `--m 17`, then
optionally once with `--m 257`. It uses BF16, hidden size 4096, `hc_mult=4`,
ordinary FP32 `fn`, no RMSNorm, `force_fused=True`, and compares all four
outputs to the separate `mhc_post` + `mhc_pre` path. It reports per-output
maximum error and per-row mismatch ratio, including the final row. A
nonfinite result or concentrated final-row mismatch is a triage signal.
Do not call the differential path an independent oracle: it shares MHC
helpers and numerical choices. Confirm any signal with the upstream
[`mhc_post_pre_ref`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_mhc.py#L735)
or a separately written Torch reference, then minimize and check support
constraints. `M=17/257` are shape **tests**, not distinct kernel types.

Neither one-GPU probe tests inter-rank MegaMoE V2 release/acquire or
forward progress. Its remote tile-ready publication, epoch wait, and
producer-residency assumptions require at least two peer GPUs; fixed-slot
V4-Pro requires eight. See [feasibility](feasibility.md). Do not simulate
multiple ranks on one GPU and label that a cross-rank result.
