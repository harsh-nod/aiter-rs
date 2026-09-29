"""CPU-side contract and independent Torch oracle for proposed OPUS cases."""

from __future__ import annotations

import json
from pathlib import Path

from .admit import BLOCK_K, BLOCK_M, BLOCK_N, PERSISTENT_KIDS, launch_geometry

SCHEMA = "aiter-rs-opus-a16w16-adversarial-candidate-v1"
AITERSHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
LAYOUT = "contiguous_[1,M,K]x[1,N,K]to[1,M,N]"
PATTERNS = {"random", "checkerboard", "cancellation", "last_k_onehot"}
INT32_MAX = 2**31 - 1
CASE_FIELDS = {"id", "batch", "m", "n", "k", "kid", "seed", "pattern", "source"}
MATRIX_FIELDS = {
    "schema", "status", "aiter_sha", "arch", "input_dtype", "output_dtype",
    "physical_layout", "split_k", "rtol", "atol", "cases",
}


def validate_case(case: dict) -> dict[str, int]:
    if set(case) != CASE_FIELDS or not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("case fields must match the public contiguous ABI")
    if case.get("batch") != 1 or case.get("kid") not in PERSISTENT_KIDS:
        raise ValueError("only batch-one exact kid 300/1300 is in this ABI")
    if case.get("pattern") not in PATTERNS or case.get("source") not in {"admitted_anchor", "proposed"}:
        raise ValueError("unknown input pattern or provenance")
    if type(case.get("seed")) is not int or case["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    m, n, k = (case[field] for field in ("m", "n", "k"))
    if any(type(value) is not int or value < 1 for value in (m, n, k)):
        raise ValueError("M/N/K must be positive integers")
    loops = (k + BLOCK_K - 1) // BLOCK_K
    if k % 2 or loops < 2 or loops % 2:
        raise ValueError("persistent K requires an even count of at least two K-tile loops")
    if n % 16:
        raise ValueError("N must be 16-aligned even with kid 300")
    if case["kid"] == 1300 and (m % BLOCK_M or n % BLOCK_N or k % BLOCK_K):
        raise ValueError("kid 1300 is no-OOB and requires full M/N/K tiles")
    if max(m * k, n * k, m * n) > INT32_MAX:
        raise ValueError("standalone adapter uses int32 dense strides")
    geometry = launch_geometry(m, n)
    if geometry["m_per_wg"] < 2:
        raise ValueError("case must execute at least two persistent M tiles per workgroup")
    return geometry


def validate_matrix(spec: dict, anchors: dict) -> None:
    if set(spec) != MATRIX_FIELDS:
        raise ValueError("public matrix fields changed or hidden data was inlined")
    if (spec.get("schema"), spec.get("status"), spec.get("aiter_sha")) != (
        SCHEMA, "proposed_public_only_not_admitted", AITERSHA,
    ):
        raise ValueError("matrix schema, status, or pinned AITER revision changed")
    if (spec.get("arch"), spec.get("input_dtype"), spec.get("output_dtype"), spec.get("physical_layout"), spec.get("split_k")) != (
        "gfx950", "bfloat16", "bfloat16", LAYOUT, 0,
    ):
        raise ValueError("matrix leaves the standalone adapter ABI")
    if spec.get("rtol") != 0.02 or spec.get("atol") != 0.125:
        raise ValueError("Torch oracle tolerance changed")
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 9 or len({c["id"] for c in cases}) != 9:
        raise ValueError("expected nine unique proposed public cases")
    anchor_cases = {case["id"]: case for case in anchors["cases"]}
    if {case["id"] for case in cases if case["source"] == "admitted_anchor"} != set(anchor_cases):
        raise ValueError("admitted anchors changed")
    for case in cases:
        validate_case(case)
        if case["source"] == "admitted_anchor":
            anchor = anchor_cases[case["id"]]
            if any(case[field] != anchor[field] for field in ("batch", "m", "n", "k", "kid", "seed")):
                raise ValueError("admitted anchor content changed")


def load_matrix(path: Path, anchors_path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    validate_matrix(spec, anchors)
    return spec


def make_operands(torch, case: dict, *, device: str, phase: int):
    """Create BF16 [1,M,K] and [1,N,K] without using AITER code."""
    if phase not in (0, 1):
        raise ValueError("phase must be zero or one")
    m, n, k = (case[field] for field in ("m", "n", "k"))
    shape_a, shape_b = (1, m, k), (1, n, k)
    pattern = case["pattern"]
    if pattern == "random":
        generator = torch.Generator(device=device).manual_seed(case["seed"] + phase)
        a = torch.randn(shape_a, generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)
        b = torch.randn(shape_b, generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)
        return a, b
    row_sign = 1 - 2 * (torch.arange(m, device=device) % 2)
    col_sign = 1 - 2 * (torch.arange(n, device=device) % 2)
    k_sign = 1 - 2 * (torch.arange(k, device=device) % 2)
    a_k = k_sign if phase == 0 else torch.ones(k, device=device)
    if pattern == "checkerboard":
        a = row_sign[:, None] * a_k[None, :]
        b = col_sign[:, None] * k_sign[None, :]
    elif pattern == "cancellation":
        a = row_sign[:, None] * a_k[None, :]
        b = torch.ones(shape_b[1:], device=device)
    else:
        a = torch.zeros(shape_a[1:], device=device)
        b = torch.zeros(shape_b[1:], device=device)
        a[:, -1 if phase == 0 else 0] = row_sign
        b[:, -1] = col_sign
    return a.unsqueeze(0).to(torch.bfloat16).contiguous(), b.unsqueeze(0).to(torch.bfloat16).contiguous()


def fp32_reference(torch, a, b):
    return torch.matmul(a.float(), b.float().transpose(-1, -2))


def compare_output(torch, actual, reference, *, atol: float = 0.125, rtol: float = 0.02) -> dict:
    actual_fp32 = actual.float()
    error = (actual_fp32 - reference).abs()
    bad = (~torch.isfinite(actual_fp32)) | (error > atol + rtol * reference.abs())
    bad_count = int(bad.sum().item())
    return {
        "pass": bad_count == 0,
        "bad_count": bad_count,
        "finite": bool(torch.isfinite(actual_fp32).all().item()),
        "max_abs_error": float(error.max().item()),
    }
