#!/usr/bin/env python3
"""One-call empty-batch GDR launch probe for a pinned gfx950 AITER checkout."""

from __future__ import annotations

import torch

from aiter.ops.gdr_decode_packed_bf16 import gdr_decode_packed_bf16


def main() -> None:
    device = torch.device("cuda")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise SystemExit(f"requires gfx950, found {arch}")

    state = torch.randn((2, 32, 128, 128), device=device, dtype=torch.bfloat16)
    before = state.clone()
    kwargs = {
        "mixed_qkv": torch.empty((0, 6144), device=device, dtype=torch.bfloat16),
        "a": torch.empty((0, 32), device=device, dtype=torch.bfloat16),
        "b": torch.empty((0, 32), device=device, dtype=torch.bfloat16),
        "dt_bias": torch.zeros((32,), device=device, dtype=torch.bfloat16),
        "A_log": torch.zeros((32,), device=device, dtype=torch.float32),
        "indices": torch.empty((0,), device=device, dtype=torch.int32),
        "state": state,
        "out": torch.empty((0, 1, 32, 128), device=device, dtype=torch.bfloat16),
    }
    try:
        out, returned_state = gdr_decode_packed_bf16(**kwargs)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"candidate: empty-batch call raised {type(exc).__name__}: {exc}")
        raise
    if out.numel() != 0 or returned_state.data_ptr() != state.data_ptr():
        raise AssertionError("unexpected output or state alias behavior")
    if not torch.equal(state.view(torch.int16), before.view(torch.int16)):
        raise AssertionError("empty-batch call mutated state")
    print("empty-batch call returned normally; state unchanged")


if __name__ == "__main__":
    main()
