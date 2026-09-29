#!/usr/bin/env python3
"""Bounded gfx950 GDR fixed-scale validation probe; no files are written."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


PINNED_SHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
WRAPPER = "aiter/ops/gdr_decode_packed_bf16.py"
KERNEL = "csrc/kernels/gdr_decode_packed_bf16.cu"


def verify_source(checkout: Path) -> None:
    sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != PINNED_SHA:
        raise SystemExit(f"expected pinned AITER {PINNED_SHA}, found {sha}")
    if subprocess.run(
        ["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--", WRAPPER, KERNEL],
        check=False,
    ).returncode:
        raise SystemExit("GDR wrapper/kernel differs from pinned source")
    wrapper = (checkout / WRAPPER).read_text()
    kernel = (checkout / KERNEL).read_text()
    if "if abs(float(scale) - expected_scale) > 1e-12:" not in wrapper:
        raise SystemExit("fixed-scale wrapper predicate changed")
    if "const float q_scale = __builtin_amdgcn_rsqf(q_sq + 1.0e-6f) * scale;" not in kernel:
        raise SystemExit("Q-scale kernel expression changed")
    expected = 128**-0.5
    if abs(float("nan") - expected) > 1e-12 or math.isfinite(float("nan")):
        raise SystemExit("local NaN comparison precondition failed")


def tensor_stats(torch, value) -> dict:
    finite = torch.isfinite(value)
    return {
        "numel": value.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(value).sum().item()),
        "inf": int(torch.isinf(value).sum().item()),
    }


def same_bf16_bits(torch, lhs, rhs) -> bool:
    return bool(torch.equal(lhs.view(torch.int16), rhs.view(torch.int16)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-checkout", type=Path, required=True)
    args = parser.parse_args()
    checkout = args.aiter_checkout.resolve()
    verify_source(checkout)
    jit_dir = os.environ.get("AITER_JIT_DIR")
    if not jit_dir or Path(jit_dir).resolve().is_relative_to(checkout):
        raise SystemExit("set AITER_JIT_DIR to a writable path outside AITER checkout")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(checkout))

    import torch
    import aiter
    from aiter.ops.gdr_decode_packed_bf16 import gdr_decode_packed_bf16

    if not Path(aiter.__file__).resolve().is_relative_to(checkout):
        raise SystemExit("imported AITER package is not from the pinned checkout")
    if not torch.cuda.is_available():
        raise SystemExit("requires a ROCm GPU")
    device = torch.device("cuda:0")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise SystemExit(f"requires gfx950, found {arch}")

    torch.manual_seed(20260929)
    mixed_qkv = (torch.randn((1, 6144), device=device) * 0.1).to(torch.bfloat16)
    initial_state = (
        torch.randn((2, 32, 128, 128), device=device) * 0.02
    ).to(torch.bfloat16)
    read_only = {
        "mixed_qkv": mixed_qkv,
        "a": torch.zeros((1, 32), device=device, dtype=torch.bfloat16),
        "b": torch.zeros((1, 32), device=device, dtype=torch.bfloat16),
        "dt_bias": torch.zeros((32,), device=device, dtype=torch.bfloat16),
        "A_log": torch.full((32,), -2.0, device=device, dtype=torch.float32),
        "indices": torch.tensor([0], device=device, dtype=torch.int32),
    }

    def fresh_call_args() -> dict:
        return {
            **read_only,
            "state": initial_state.clone(),
            "out": torch.full(
                (1, 1, 32, 128), -99.0, device=device, dtype=torch.bfloat16
            ),
        }

    result = {
        "probe": "gdr_fixed_scale_nan_v1",
        "aiter_sha": PINNED_SHA,
        "arch": arch,
        "source_basis": "wrapper fixed-scale predicate at L68 and Q-scale multiply at kernel L202",
        "classification": "unclassified",
    }

    bad = fresh_call_args()
    bad_state_before, bad_out_before = bad["state"].clone(), bad["out"].clone()
    try:
        gdr_decode_packed_bf16(**bad, scale=1.0)
        torch.cuda.synchronize()
        bad_exception = None
    except Exception as exc:
        bad_exception = type(exc).__name__
    result["scale_1"] = {
        "exception_type": bad_exception,
        "state_unchanged": same_bf16_bits(torch, bad["state"], bad_state_before),
        "out_unchanged": same_bf16_bits(torch, bad["out"], bad_out_before),
    }
    if bad_exception != "ValueError" or not all(
        result["scale_1"][key] for key in ("state_unchanged", "out_unchanged")
    ):
        result["classification"] = "control_failed_inconclusive"
        print(json.dumps(result, indent=2))
        return 2

    valid = fresh_call_args()
    try:
        valid_out, valid_state = gdr_decode_packed_bf16(**valid)
        torch.cuda.synchronize()
    except Exception as exc:
        result["classification"] = "valid_control_failed_inconclusive"
        result["valid_control_exception"] = type(exc).__name__
        print(json.dumps(result, indent=2))
        return 2
    result["valid_control"] = {
        "out": tensor_stats(torch, valid_out),
        "state": tensor_stats(torch, valid_state),
        "return_aliases": (
            valid_out.data_ptr() == valid["out"].data_ptr()
            and valid_state.data_ptr() == valid["state"].data_ptr()
        ),
        "untouched_slot_unchanged": same_bf16_bits(
            torch, valid_state[1], initial_state[1]
        ),
    }
    if (
        result["valid_control"]["out"]["finite"] != valid_out.numel()
        or result["valid_control"]["state"]["finite"] != valid_state.numel()
        or not result["valid_control"]["return_aliases"]
        or not result["valid_control"]["untouched_slot_unchanged"]
    ):
        result["classification"] = "valid_control_failed_inconclusive"
        print(json.dumps(result, indent=2))
        return 2

    nan_args = fresh_call_args()
    nan_state_before, nan_out_before = nan_args["state"].clone(), nan_args["out"].clone()
    try:
        nan_out, nan_state = gdr_decode_packed_bf16(**nan_args, scale=float("nan"))
        torch.cuda.synchronize()
    except Exception as exc:
        result["nan_scale"] = {
            "accepted": False,
            "exception_type": type(exc).__name__,
            "out": tensor_stats(torch, nan_args["out"]),
            "state": tensor_stats(torch, nan_args["state"]),
            "out_unchanged": same_bf16_bits(torch, nan_args["out"], nan_out_before),
            "state_unchanged": same_bf16_bits(torch, nan_args["state"], nan_state_before),
        }
        clean_rejection = (
            isinstance(exc, ValueError)
            and result["nan_scale"]["out_unchanged"]
            and result["nan_scale"]["state_unchanged"]
        )
        result["classification"] = (
            "nan_rejected" if clean_rejection else "nan_error_inconclusive"
        )
        print(json.dumps(result, indent=2))
        return 0 if clean_rejection else 2

    result["nan_scale"] = {
        "accepted": True,
        "out": tensor_stats(torch, nan_out),
        "state": tensor_stats(torch, nan_state),
        "return_aliases": (
            nan_out.data_ptr() == nan_args["out"].data_ptr()
            and nan_state.data_ptr() == nan_args["state"].data_ptr()
        ),
        "state_bitwise_equal_to_valid_control": same_bf16_bits(
            torch, nan_state, valid_state
        ),
        "untouched_slot_unchanged": same_bf16_bits(
            torch, nan_state[1], initial_state[1]
        ),
    }
    result["classification"] = "candidate_fixed_scale_validation_bypass_contract_pending"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
