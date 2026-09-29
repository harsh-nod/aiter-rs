#!/usr/bin/env python3
"""Reproduce gfx950 OPUS persistent BF16 partial-K behavior with AITER only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


M = 8192
N = 4096
K_VALUES = (194, 254, 256)
SEEDS = {194: 95114, 254: 95115, 256: 95116}
KID = 300
GUARD = 512
ATOL = 0.5
RTOL = 0.03
SOURCE_PATHS = (
    "aiter/ops/opus/gemm_op_a16w16.py",
    "csrc/opus_gemm/opus_gemm_tune.py",
    "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
    "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh",
)


def k_loop_count(k: int) -> int:
    return (k + 63) // 64


def check_cases() -> None:
    for k in K_VALUES:
        loops = k_loop_count(k)
        if loops < 2 or loops % 2:
            raise ValueError(f"K={k} is outside the persistent even-loop domain")
    if N % 16:
        raise ValueError("N must be 16-aligned for persistent kid 300")


def source_revision(checkout: Path) -> str:
    sha = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--", *SOURCE_PATHS],
        check=False,
    ).returncode
    if changed:
        raise SystemExit("AITER OPUS wrapper/tuner/launcher/pipeline differs from checkout HEAD")
    return sha


def classify(output: dict) -> tuple[bool, bool]:
    controls = [
        *output["contiguous"].values(),
        output["k194_padded_row"]["zero_padding"],
        output["k194_padded_row"]["one_padding"],
    ]
    guards = all(r["guards_intact"] and r["inputs_unchanged"] and r["padding_unchanged"] for r in controls)
    observed = bool(
        guards
        and output["contiguous"]["194"]["bad_elements"] > 0
        and output["contiguous"]["254"]["bad_elements"] > 0
        and output["contiguous"]["256"]["bad_elements"] == 0
        and output["k194_padded_row"]["zero_padding"]["bad_elements"] == 0
        and output["k194_padded_row"]["one_padding"]["bad_elements"] > 0
        and output["k194_padded_row"]["outputs_differ"]
    )
    return guards, observed


def guarded_tensor(torch, shape: tuple[int, ...], sentinel: float, fill: float):
    count = 1
    for extent in shape:
        count *= extent
    storage = torch.full((count + 2 * GUARD,), sentinel, dtype=torch.bfloat16, device="cuda")
    value = storage[GUARD : GUARD + count].view(shape)
    value.fill_(fill)
    return storage, value


def guard_intact(torch, storage, sentinel: float) -> bool:
    return bool(torch.all(storage[:GUARD] == sentinel).item() and torch.all(storage[-GUARD:] == sentinel).item())


def operands(torch, k: int):
    generator = torch.Generator(device="cpu").manual_seed(SEEDS[k])
    a = torch.randn((1, M, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((1, N, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    return a.to("cuda"), b.to("cuda")


def compare(torch, actual, reference) -> dict:
    values = actual.float()
    finite = torch.isfinite(values)
    error = torch.abs(values - reference)
    bad = ~finite | (error > ATOL + RTOL * torch.abs(reference))
    return {
        "bad_elements": int(bad.sum().item()),
        "total_elements": actual.numel(),
        "nonfinite_elements": int((~finite).sum().item()),
        "max_abs_error": float(error.max().item()) if bool(torch.all(finite).item()) else None,
    }


def run_one(torch, opus_bmm, logical_a, logical_b, reference, *, pad_to: int | None = None, pad_value: float = 0.0):
    k = logical_a.shape[-1]
    physical_k = k if pad_to is None else pad_to
    if physical_k < k:
        raise ValueError("physical K stride must contain logical K")
    a_storage, a_full = guarded_tensor(torch, (1, M, physical_k), 13.0, pad_value)
    b_storage, b_full = guarded_tensor(torch, (1, N, physical_k), -17.0, pad_value)
    y_storage, y = guarded_tensor(torch, (1, M, N), 23.0, -99.0)
    a = a_full[:, :, :k]
    b = b_full[:, :, :k]
    a.copy_(logical_a)
    b.copy_(logical_b)

    opus_bmm(a, b, y, kid=KID, split_k=0)
    torch.cuda.synchronize()
    result = {
        **compare(torch, y, reference),
        "a_stride": list(a.stride()),
        "b_stride": list(b.stride()),
        "inputs_unchanged": bool(torch.equal(a, logical_a) and torch.equal(b, logical_b)),
        "padding_unchanged": bool(
            torch.all(a_full[:, :, k:] == pad_value).item()
            and torch.all(b_full[:, :, k:] == pad_value).item()
        ),
        "guards_intact": all((
            guard_intact(torch, a_storage, 13.0),
            guard_intact(torch, b_storage, -17.0),
            guard_intact(torch, y_storage, 23.0),
        )),
    }
    return result, y.clone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-checkout", type=Path, required=True)
    args = parser.parse_args()
    checkout = args.aiter_checkout.resolve()
    if not (checkout / "aiter/ops/opus/gemm_op_a16w16.py").is_file():
        parser.error("--aiter-checkout is not an AITER source checkout")
    sha = source_revision(checkout)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(checkout))

    import torch
    import aiter
    from aiter.ops.opus import opus_bmm

    if not Path(aiter.__file__).resolve().is_relative_to(checkout):
        raise SystemExit("imported AITER is not from --aiter-checkout")
    if not torch.cuda.is_available():
        raise SystemExit("requires a ROCm GPU")
    arch = torch.cuda.get_device_properties("cuda:0").gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise SystemExit(f"requires gfx950, found {arch}")
    check_cases()

    output = {
        "probe": "opus_persistent_partial_k_aiter_only_v1",
        "aiter_sha": sha,
        "arch": arch,
        "kid": KID,
        "shape_mn": [M, N],
        "tolerance": {"atol": ATOL, "rtol": RTOL, "reference": "Torch FP32 matmul"},
        "contiguous": {},
        "k194_padded_row": {},
    }
    for k in K_VALUES:
        logical_a, logical_b = operands(torch, k)
        reference = torch.matmul(logical_a.float(), logical_b.float().transpose(-1, -2))
        result, _ = run_one(torch, opus_bmm, logical_a, logical_b, reference)
        output["contiguous"][str(k)] = result
        if k == 194:
            zero, zero_y = run_one(torch, opus_bmm, logical_a, logical_b, reference, pad_to=256, pad_value=0.0)
            one, one_y = run_one(torch, opus_bmm, logical_a, logical_b, reference, pad_to=256, pad_value=1.0)
            output["k194_padded_row"] = {
                "zero_padding": zero,
                "one_padding": one,
                "outputs_differ": not bool(torch.equal(zero_y, one_y)),
            }

    all_guards, observed = classify(output)
    output["guards_and_inputs_pass"] = all_guards
    output["partial_k_padding_dependence_observed"] = observed
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all_guards else 2


if __name__ == "__main__":
    raise SystemExit(main())
