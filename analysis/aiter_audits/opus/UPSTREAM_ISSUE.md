Filed as [ROCm/aiter #5954](https://github.com/ROCm/aiter/issues/5954).

## Summary

On MI350X/gfx950, `opus_bmm` with explicit persistent BF16 `kid=300` accepts partial K values such as 194 and 254, but produces silently incorrect outputs. For K=194, changing only bytes in the *physical row padding* of A and B changes every output even though the logical tensors are identical. K=256 is correct. This was reproduced directly through pinned AITER, without an external kernel adapter.

## Reproducer

Run the [standalone AITER-only probe](https://github.com/harsh-nod/aiter-rs/blob/32cf3b6/analysis/aiter_audits/opus/probe_partial_k.py) on gfx950 with a clean AITER checkout and writable JIT cache:

```sh
git clone https://github.com/harsh-nod/aiter-rs.git
PYTHONPATH=/path/to/aiter AITER_JIT_DIR=/tmp/aiter-jit-cache \
  python3 aiter-rs/analysis/aiter_audits/opus/probe_partial_k.py \
  --aiter-checkout /path/to/aiter
```

It calls `aiter.ops.opus.opus_bmm(A, B, Y, kid=300, split_k=0)` with BF16 A `[1,8192,K]` and B `[1,4096,K]`, and compares the complete BF16 output against Torch FP32 matmul at the public OPUS test tolerance `atol=0.5, rtol=0.03`. Inputs and output have outer canaries; the probe also checks input and padding immutability.

Observed on AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`, MI350X/gfx950:

| Input | Incorrect / 33,554,432 outputs | Max absolute error |
| --- | ---: | ---: |
| Contiguous K=194 | 30,689,090 | 47.108 |
| Contiguous K=254 | 14,448,196 | 18.077 |
| Contiguous K=256 | 0 | 0.250 |
| Logical K=194, physical row stride 256, padding 0 | 0 | 0.249 |
| Same logical K=194 values and stride, padding 1 | 33,554,432 | 62.452 |

All input/output canaries and input/padding immutability checks passed. In the last two rows, **only the padding beyond logical K changed**. The JSON result and full provenance are summarized in the [study receipt](https://github.com/harsh-nod/aiter-rs/blob/32cf3b6/analysis/aiter_audits/opus/README.md); private raw JSON SHA256 is `43740d5266a808d14948adc02f863128e4f3e5434bbcd8f5d0c7bcbfc5ed7ea0`.

## Expected behavior and suspected boundary

The result should depend only on logical A and B and agree with the matmul reference within tolerance, or the wrapper/tuner should reject an unsupported K before launch. The pinned tuner admits kid 300 for even `ceil_div(K,64)` and N divisible by 16; it does not require K divisible by 64. The persistent pipeline uses ceil-div K loops and 64-wide async A/B loads. The padding intervention strongly suggests that the partial-K path consumes physical row padding, but it does not identify the exact device instruction.

The relevant OPUS wrapper, tuner, generated launcher, device header, and public test file were byte-identical between the pinned commit and `main` at `9ef0d08fc20c81755c30c4f867b2f39664c75702` (2026-09-29). **Runtime behavior was tested at the pinned commit only**, not on current main.

This is a pinned AITER issue, not an agent-written-kernel failure. A public one-hot K=194 fixture had passed earlier; fresh random operands and the zero/one padding intervention exposed the defect.
