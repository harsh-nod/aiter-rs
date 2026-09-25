# gfx950 MXFP4 quantization

Implement the exported C ABI in the supplied `starter.hip`. Your kernel
quantizes a contiguous FP16 or BF16 matrix `X[M,K]` on gfx950, with one
MXFP4 E2M1 scale for each contiguous group of 32 values along `K`. Aim to
match the pinned AITER operator's bytes and latency on the supplied GPU.

The scorer supplies distinct, 16-byte-aligned device buffers for input,
packed FP4 output (`M*K/2` bytes), and E8M0 scales (`M*K/32` bytes). `dtype=0`
is FP16, `dtype=1` is BF16; `M>=1`, `K>=32` and divisible by 32. The task
uses `round_mode=Even` and no shuffles. Values are finite normal values or
signed zero with magnitude at most 128. Outputs must be fully written; the
input and bytes outside output spans must not change. There is no allowed
persistent state. Enqueue on the provided stream, return a HIP status, and
do not synchronize.

For each group, convert values to FP32 and take `amax=max(abs(x))`. Compute:

```text
r_bits = (bits_f32(amax) + 0x00200000) & 0x7F800000
exponent = -127 if r_bits == 0 else clamp(exponent_bits(r_bits)-129, -127, 127)
scale = 2^exponent
scale_byte = exponent+127
```

Divide each value by `scale`, round to nearest E2M1 with ties to the even
code, and saturate at magnitude 6. The positive levels by code are
`0, 0.5, 1, 1.5, 2, 3, 4, 6`; bit 3 is the sign (including negative zero).
Pack even-indexed values into low nibbles and odd-indexed values into high
nibbles. Scale bytes are row-major, one per 32 input values.

You may edit `starter.hip` and run the visible correctness/benchmark tests.
The submitted source must compile with the provided HIP compiler, use the
specified ABI, and contain the implementation itself; do not call AITER,
PyTorch, external quantization libraries, or precomputed outputs. The
submitted final candidate is evaluated on additional unseen cases. Record
your timing and correctness observations as you iterate.
