# Pilot protocol: gfx950 MXFP4 Even, no shuffle

Status: **candidate only, not scored-eligible**. No AITER baseline, independent
oracle, HIP parity route, or measurement noise has yet been validated on the
target GPU. Do not launch scored agents or count this as a distinct eligible
type until the admission checklist at the end passes.

## Identity and pinned dispatch

- Task ID: `gfx950.quant_mxfp4.even.noshuffle.v1`.
- AITER revision: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- User-facing call: [`quant_mxfp4_hip`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/quant.py#L945)
  with `group_size=32`, `round_mode=2` (`Even`), and all shuffle/gate flags
  `False`. Because the fast-path condition tests `RoundUp`, `Even` reaches
  the [`quant_mxfp4` native binding](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/quant.py#L931).
- Native dispatch: [`csrc/kernels/quant_mxfp4.cu`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/quant_mxfp4.cu#L238)
  selects `MxScaleRoundMode::Even` and launches `quant_mxfp4_kernel` with
  FP16/BF16 dtype specialization. On gfx950, packing uses
  `__builtin_amdgcn_cvt_scalef32_pk_fp4_{f16,bf16}` (`__gfx950__` branch).
- Run only on an observed `gfx950` target. Record exact GPU SKU, ROCm and
  compiler versions, source SHA, launch path, and device topology in the
  result. The expected machine is `mi350-2`; its observed SKU must be logged.

## Supported domain for this pilot

Input is one 16-byte-aligned, contiguous GPU tensor `X[M,K]` of FP16 or BF16.
`1 <= M <= 4097`, `32 <= K <= 2048`, and `K` is divisible by 32. Values are
finite normal numbers or signed zero, with absolute value at most 128. Input,
packed output, and scale output do not overlap. This bounded domain is a
*pilot* contract; it does not claim to cover every input accepted by AITER.
Subnormals, NaN/Inf, noncontiguous tensors, zero dimensions, alternate round
modes, and shuffle flags require separately vetted task variants.

The scorer preallocates distinct contiguous outputs: `P[M,K/2]` raw packed
FP4 E2M1 bytes and `S[M,K/32]` raw E8M0 scale bytes. The agent must write
every output byte and leave input and bytes outside each output span
unchanged. There are no persistent effects or allowed cross-call state. The
public Python wrapper may use PyTorch's FP4/E8M0 dtypes, but byte comparison
uses their `uint8` views. Aliasing is **not** a supported input configuration.

For each 32-value group, let `amax=max(abs(x_i))` in FP32. Even-mode scale is
defined by AITER's `round_pow2_1.75` bit rule:

```text
r_bits = (bits_f32(amax) + 0x00200000) & 0x7F800000
if r_bits == 0: exponent = -127
else:           exponent = clamp(exponent_bits(r_bits) - 129, -127, 127)
scale = 2^exponent
S[group] = exponent + 127
```

Each input value is divided by `scale`, rounded to the nearest FP4 E2M1
level with ties to the even code, saturated to magnitude 6, and packed with
the even-indexed element in the low nibble and odd-indexed element in the
high nibble. Preserve the sign bit of negative zero. The unsigned positive
levels by code are `0, 0.5, 1, 1.5, 2, 3, 4, 6`; bit 3 is the sign. This
description is the agent-facing mathematical contract, not a promise that
the independently implemented oracle and AITER have already agreed at all
thresholds. Resolve any disagreement before scoring.

## C ABI and agent workspace

Use [`references/quant_mxfp4_abi.h`](../../references/quant_mxfp4_abi.h)
after the harness branch is integrated:

```c
extern "C" int aiter_rs_quant_mxfp4_even(
    const void* input, void* packed_output, void* scale_output,
    int64_t rows, int64_t cols, int dtype, void* stream);
```

`dtype=0` means FP16; `dtype=1` means BF16. Pointers address device buffers.
The function must enqueue its work on the supplied HIP stream, return a
`hipError_t` integer status, and not synchronize. The scorer owns all
allocations and reports launch failures; compilation/JIT setup is outside
the timed region. The visible agent workspace should contain only the prompt,
ABI header, starter source, visible fixtures, and toolchain instructions;
do not mount the AITER implementation, this private spec, or hidden cases.
If public-source lookup cannot be blocked, record it as exposure and analyze
that run separately.

## Oracle, cases, and timing

The independent oracle must implement the scale bit rule and E2M1 nearest-
even quantizer separately from AITER source and tests. The harness draft is
[`references/quant_mxfp4.py`](../../references/quant_mxfp4.py); the upstream
[`test_quant_mxfp4.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_quant_mxfp4.py#L163)
provides cross-check cases, not the sole oracle. For every scored case,
require **byte equality** of both outputs against the independent oracle
*and* pinned AITER. If the two baselines disagree, quarantine that case as a
baseline/contract problem rather than assign an agent bug. Include canary
checks and an input checksum to detect out-of-bounds or input writes.

Proposed visible correctness cases (not frozen): FP16 `(3,96)` with threshold
and signed-zero values; FP16 `(128,1024)` random; BF16 `(256,2048)` random.
Proposed hidden cases: BF16 `(7,160)` with threshold/signed-zero values,
FP16 `(1,32)` with an all-zero group, BF16 `(125,64)` with mixed zero/nonzero
groups, and BF16 `(4097,256)` for the final partial grid. Input seeds and
distributions are frozen before trials and stored only in the harness.
Add adversarial ties at all FP4 midpoints after quantization, scale
transition boundaries, alternating signs, and repeated calls with new
output buffers. The visible set must exercise both dtypes and a performance
bucket; hidden cases must not be exposed through logs or filenames.

Benchmark only byte-correct candidates. Required performance buckets are
provisionally FP16 `(128,1024)` and BF16 `(256,2048)`; add a larger bandwidth
bucket after feasibility review if this pair misses the interesting regime.
Compare the agent launch to AITER's **low-level** `quant_mxfp4` call with
preallocated `P` and `S`, identical inputs and stream, all flags off. Do not
time high-level `quant_mxfp4_hip` allocations on only one side. Qualify
same-stream event or graph-replay timing, warmup, sample count, clocks,
interference, and median absolute deviation before freezing the metric.
The plan's default non-inferiority target is median HIP latency <= 1.05x
AITER in **each** required bucket; never change it after observing agent
results. Record individual latency samples, variance, both output checks,
and functional/performance/joint pass separately. Retain every failed
candidate in the error corpus.

## Admission checklist

1. Run pinned AITER `quant_mxfp4_hip` on `mi350-2`, confirm the `Even`
   native path and exact observed gfx950 SKU/ROCm build.
2. Validate the independent CPU oracle against AITER byte-for-byte for
   public, hidden, and adversarial cases. Investigate disagreements.
3. Compile the C ABI starter and a non-agent HIP reference with the actual
   toolchain, check both dtypes, side effects, and sanitizer/canary behavior.
4. Benchmark AITER and reference at the same preallocated-output boundary,
   prove measurement noise is below the frozen threshold, and establish a
   credible HIP route to parity. Correct-but-slower reference is not proof
   of parity feasibility.
5. Freeze cases, seeds, ABI, visible prompt, tool access, threshold, and
   task hash. Only then set `scored_eligible=true` and start agent runs.
