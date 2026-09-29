# OPUS persistent partial-K source-domain investigation

**AITER-only causal control, not an agent finding.** In a public nine-case
fixture, K=194 with a last-K one-hot input passed pinned AITER and an
unchanged-source adapter. A later random-input graph control failed both
paths identically. We reran fresh random BF16 operands at fixed
`(M,N)=(8192,4096)`, kid 300, and checked every BF16 output against
`torch.matmul(A.float(), B.float().transpose(-1,-2))` with the existing
`atol=0.125, rtol=0.02` gate. Inputs and guards remained unchanged.

| Logical K | Domain/result | AITER and adapter bad elements / 33,554,432 | Max absolute error |
| ---: | --- | ---: | ---: |
| 192 | Rejected before launch: `ceil_div(K,64)=3` is odd | n/a | n/a |
| 194 | Four-loop kid 300 launch; both paths bitwise identical | 32,350,215 | 51.468 |
| 254 | Four-loop kid 300 launch; both paths bitwise identical | 23,434,451 | 18.827 |
| 256 | Four-loop kid 300 launch; six guarded/reuse steps passed | 0 | 0.250 |

For causal isolation, the AITER wrapper (not the standalone adapter ABI)
was given the **same logical K=194 operands** as a view into rows of physical
stride 256: A stride `[2097152,256,1]`, B stride `[1048576,256,1]`.
With physical padding set to 0.0, all 33,554,432 outputs passed both the
FP32 and BF16-cast Torch oracles (maximum absolute error 0.249/0.250).
Changing *only* the physical padding to 1.0 made all 33,554,432 outputs
fail both oracles (maximum absolute error about 62.5); outputs changed,
while logical operands, padding values, and input/output guards remained
intact. This is strong evidence that the partial-K path consumes physical
row padding beyond logical K, not merely a BF16 tolerance effect or an
adapter-specific ABI bug. It is not an exhaustive hardware-level
instruction or memory-safety proof.

The pinned [persistent tuner](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/opus_gemm_tune.py#L698)
admits kid 300 when the 64-wide K-loop count is even and N is 16-aligned;
unlike no-OOB kid 1300, it does not require K divisible by 64. The
[device pipeline](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L196)
uses `ceil_div(k, B_K)` loops and issues 64-wide async A/B loads; the
[AITER wrapper](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/opus/gemm_op_a16w16.py#L89)
permits K-contiguous operands with padded row strides. These source facts
and the padding intervention support a partial-K load/predicate defect,
but the exact offending instruction is not yet profiler- or ISA-proven.

**Disposition:** exclude K=194, K=254, and partial-K generally from this
task's supported/scored domain pending a source-level fix and independent
re-admission. Preserve the historical nine-case matrix and its one-hot pass
as provenance; do not count the failure as agent-error incidence. Do not
file an upstream issue before the independent receipt/reproducer review.
There were no private inputs and no latency measurements in this repro.

Raw provenance: pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`; source-equivalent adapter
binary SHA256
`356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50`;
repro driver raw SHA256
`c91c9d6faf0353498ec692cb83297d65d27ba75bd344c761a513561d323806e5`;
private raw result SHA256
`ccca6abffed0ec4f0596c0abb9f59c58cf03314a868f25a0a087674334008cbd`;
private host GPU identity report SHA256
`4b7b8c1d2bf204b465422b93c182fdb8cf4aa478bb47ecf2494380b05d364d4c`.
