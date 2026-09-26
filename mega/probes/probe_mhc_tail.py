#!/usr/bin/env python3
"""One-call MHC fused/unfused tail differential; triage only, not an oracle."""

from __future__ import annotations

import argparse
import json

import torch

from aiter.ops.mhc import mhc_fused_post_pre, mhc_post, mhc_pre


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(17, 257), default=17)
    args = parser.parse_args()

    device = torch.device("cuda")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise SystemExit(f"requires gfx950, found {arch}")
    torch.manual_seed(20260925 + args.m)
    m, hidden, mult = args.m, 4096, 4
    layer = torch.randn((m, hidden), device=device, dtype=torch.bfloat16) * 0.1
    residual = torch.randn((m, mult, hidden), device=device, dtype=torch.bfloat16) * 0.1
    post_layer_mix = torch.randn((m, mult, 1), device=device, dtype=torch.float32) * 0.1
    comb_res_mix = torch.randn((m, mult, mult), device=device, dtype=torch.float32) * 0.1
    fn = torch.randn((24, mult * hidden), device=device, dtype=torch.float32) * 0.02
    hc_scale = torch.randn((3,), device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn((24,), device=device, dtype=torch.float32) * 0.1
    options = {"hc_post_mult_value": 2.0, "sinkhorn_repeat": 20}

    fused = mhc_fused_post_pre(
        layer, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        force_fused=True, **options,
    )
    next_residual = torch.empty_like(residual)
    mhc_post(next_residual, layer, residual, post_layer_mix, comb_res_mix)
    pre = mhc_pre(next_residual, fn, hc_scale, hc_base, **options)
    unfused = (*pre, next_residual)
    torch.cuda.synchronize()

    result = {"m": m, "arch": arch, "comparison": "fused versus unfused HIP, triage only"}
    for name, actual, expected in zip(
        ("post_mix", "comb_mix", "layer_input", "next_residual"), fused, unfused
    ):
        a, b = actual.float(), expected.float()
        close = torch.isclose(a, b, rtol=1e-2, atol=1e-2)
        mismatch_per_row = (~close).reshape(m, -1).float().mean(dim=1)
        result[name] = {
            "mismatch_fraction": float((~close).float().mean().item()),
            "last_row_mismatch_fraction": float(mismatch_per_row[-1].item()),
            "worst_row_mismatch_fraction": float(mismatch_per_row.max().item()),
            "max_abs_error": float((a - b).abs().max().item()),
            "finite": bool(torch.isfinite(a).all().item()),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
