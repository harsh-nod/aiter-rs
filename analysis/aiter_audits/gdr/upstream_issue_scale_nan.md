## Summary

`gdr_decode_packed_bf16` documents/enforces a fixed Q scale, but its check
accepts `scale=float("nan")`. On MI350X/gfx950 the call returns an all-NaN
output instead of rejecting the invalid scale. The recurrent state stays
finite and matched the normal-scale control bitwise in the bounded probe.

## Reproduction

```python
import torch
from aiter.ops.gdr_decode_packed_bf16 import gdr_decode_packed_bf16

device = "cuda"
args = dict(
    mixed_qkv=torch.randn((1, 6144), device=device).to(torch.bfloat16),
    a=torch.zeros((1, 32), device=device, dtype=torch.bfloat16),
    b=torch.zeros((1, 32), device=device, dtype=torch.bfloat16),
    dt_bias=torch.zeros((32,), device=device, dtype=torch.bfloat16),
    A_log=torch.full((32,), -2.0, device=device),
    indices=torch.tensor([0], device=device, dtype=torch.int32),
    state=torch.randn((2, 32, 128, 128), device=device).to(torch.bfloat16),
    out=torch.empty((1, 1, 32, 128), device=device, dtype=torch.bfloat16),
)
output, state = gdr_decode_packed_bf16(**args, scale=float("nan"))
torch.cuda.synchronize()
print(torch.isnan(output).sum().item(), output.numel())
```

Observed: `4096 4096`. A control call with `scale=1.0` raises `ValueError`;
the default fixed scale produces finite output. In the fuller controlled run,
the NaN-scale state was all finite, equal bitwise to the default-scale state,
and an untouched state slot remained unchanged.

## Cause and expected behavior

The [wrapper's check](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L65-L69)
uses `abs(float(scale) - expected_scale) > 1e-12`. That comparison is false
for NaN, which then reaches the [Q normalization](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L202).
Please reject nonfinite scales under the fixed-scale contract before launch.
This is a Python validation issue, not a claim of a GDR synchronization or
state-update bug.

Runtime pin: AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`, MI350X
gfx950, no-network ROCm image ID
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
The same comparison remains on AITER `main` at
`2b6ff3d6b5bd20ea7197bfb6d7307e4a235904a2` (checked 2026-09-29).
A [bounded probe and receipt](https://github.com/harsh-nod/aiter-rs/tree/main/analysis/aiter_audits/gdr)
include the finite control and state checks.
