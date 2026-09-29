"""Pinned, correctness-only gfx950 AITER sparse-prefill admission probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import secrets
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from references.sparse_prefill_gfx950 import AITER_SHA, Case, make_inputs, oracle, public_cases


TASK_ID = "sparse-prefill-gfx950-bf16-hle32"
GUARD = 128
RTOL = ATOL = 1e-2
MAX_MISMATCH_FRACTION = 0.01


def private_cases() -> tuple[Case, ...]:
    return (
        Case(secrets.token_hex(8), secrets.randbits(63), 7, 16, 80, 96, "mixed"),
        Case(secrets.token_hex(8), secrets.randbits(63), 5, 32, 96, 80, "tile_tail"),
        Case(secrets.token_hex(8), secrets.randbits(63), 4, 17, 96, 80, "mixed"),
        Case(secrets.token_hex(8), secrets.randbits(63), 4, 32, 80, 80, "dense", 0.03125),
    )


def generate_private(path: Path) -> tuple[str, int]:
    cases = private_cases()
    for case in cases:
        case.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    matrix = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "aiter_sha": AITER_SHA,
        "generator": "references.sparse_prefill_admission.private_cases",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [asdict(case) for case in cases],
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(matrix, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest(), len(cases)


def load_cases(path: Path | None) -> tuple[Case, ...]:
    if path is None:
        return public_cases()
    matrix = json.loads(path.read_text())
    if matrix.get("schema_version") != 1 or matrix.get("task_id") != TASK_ID:
        raise ValueError("unrecognized private matrix")
    if matrix.get("aiter_sha") != AITER_SHA:
        raise ValueError("private matrix AITER revision mismatch")
    if matrix.get("generator_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("private matrix generator revision mismatch")
    cases = tuple(Case(**item) for item in matrix["cases"])
    for case in cases:
        case.validate()
    return cases


def _probe_case(case: Case, op) -> dict:
    cpu = make_inputs(case)
    expected = oracle(cpu, case.softmax_scale)
    inputs = {key: value.cuda() for key, value in cpu.items()}
    storage = torch.full(
        (expected.numel() + 2 * GUARD,), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    out = storage[GUARD:-GUARD].view_as(expected)
    low_level_calls = {"gfx950": 0, "gfx1250": 0}
    original_950 = op.pa_sparse_prefill_gfx950_opus_fwd
    original_1250 = op.pa_sparse_prefill_gfx1250_opus_fwd

    def counted_950(*args, **kwargs):
        low_level_calls["gfx950"] += 1
        return original_950(*args, **kwargs)

    def counted_1250(*args, **kwargs):
        low_level_calls["gfx1250"] += 1
        return original_1250(*args, **kwargs)

    op.pa_sparse_prefill_gfx950_opus_fwd = counted_950
    op.pa_sparse_prefill_gfx1250_opus_fwd = counted_1250
    iterations = []
    try:
        for _ in range(2):
            out.fill_(7.0)
            before = low_level_calls.copy()
            returned = op.pa_sparse_prefill_opus(
                **inputs, softmax_scale=case.softmax_scale, out=out
            )
            torch.cuda.synchronize()
            actual = out.cpu()
            close = torch.isclose(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)
            mismatch_fraction = float((~close).float().mean())
            route = {key: low_level_calls[key] - before[key] for key in low_level_calls}
            guards_ok = bool((storage[:GUARD] == 7).all() and (storage[-GUARD:] == 7).all())
            inputs_ok = all(torch.equal(value.cpu(), cpu[key]) for key, value in inputs.items())
            finite = bool(torch.isfinite(actual).all())
            nontrivial = not bool(expected.abs().any()) or bool(actual.abs().any())
            passed = (
                returned is out
                and route == {"gfx950": 1, "gfx1250": 0}
                and mismatch_fraction <= MAX_MISMATCH_FRACTION
                and finite and guards_ok and inputs_ok and nontrivial
            )
            iterations.append({
                "passed": passed,
                "route": route,
                "mismatch_fraction": mismatch_fraction,
                "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                "guards_ok": guards_ok,
                "inputs_unchanged": inputs_ok,
                "finite": finite,
                "nontrivial": nontrivial,
            })
    finally:
        op.pa_sparse_prefill_gfx950_opus_fwd = original_950
        op.pa_sparse_prefill_gfx1250_opus_fwd = original_1250
    return {"name": case.name, "passed": all(item["passed"] for item in iterations), "iterations": iterations}


def probe(aiter_source: Path, matrix_path: Path | None) -> dict:
    pinned = subprocess.check_output(
        ["git", "-C", str(aiter_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if pinned != AITER_SHA:
        raise RuntimeError(f"AITER checkout is {pinned}, expected {AITER_SHA}")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"expected gfx950, found {arch}")
    sys.path.insert(0, str(aiter_source))
    op = importlib.import_module("aiter.ops.pa_sparse_prefill_opus")
    if Path(op.__file__).resolve() != (aiter_source / "aiter/ops/pa_sparse_prefill_opus.py").resolve():
        raise RuntimeError("imported AITER module is not from the pinned checkout")
    if op.get_gfx_runtime() != "gfx950":
        raise RuntimeError("AITER dispatch does not report gfx950")
    private = matrix_path is not None
    cases = load_cases(matrix_path)
    findings = [_probe_case(case, op) for case in cases]
    result = {
        "task_id": TASK_ID,
        "aiter_sha": AITER_SHA,
        "arch": arch,
        "case_count": len(findings),
        "passed_count": sum(item["passed"] for item in findings),
        "all_passed": all(item["passed"] for item in findings),
        "performance": "not_run",
        "agent_trials": 0,
        "private": private,
    }
    if not private:
        result["cases"] = findings
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-private", type=Path)
    parser.add_argument("--aiter-source", type=Path)
    parser.add_argument("--matrix", type=Path, help="private matrix, read by trusted probe only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.generate_private:
        digest, count = generate_private(args.generate_private)
        print(json.dumps({"task_id": TASK_ID, "case_count": count, "sha256": digest}))
        return 0
    if not args.aiter_source or not args.output:
        parser.error("--aiter-source and --output are required for a probe")
    result = probe(args.aiter_source, args.matrix)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
