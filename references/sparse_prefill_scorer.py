"""Guarded, correctness-only HIP candidate scorer for BF16 sparse prefill.

Run under an external process watchdog. This scorer is not a security sandbox
for arbitrary native code and does not benchmark or establish scored eligibility.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from references.sparse_prefill_admission import (
    ATOL,
    GUARD,
    MAX_MISMATCH_FRACTION,
    RTOL,
    TASK_ID,
    load_cases,
)
from references.sparse_prefill_gfx950 import AITER_SHA, Case, make_inputs, oracle


PRIVATE_MATRIX_SHA256 = "70feede754e3f5c18d0f1cbbf3f6434b6602cd97cab4e3d6b6bc11073bfcdebd"
ENTRYPOINT = "sparse_prefill_bf16_gfx950"
POISON = 7.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_outside_repo(path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    if path.resolve().is_relative_to(repo):
        raise ValueError("private matrix and scorer outputs must remain outside the repository")


class HipCandidate:
    """Thin ctypes binding to the candidate C ABI; no allocation or sync."""

    def __init__(self, path: Path):
        if path.suffix != ".so" or not path.is_file():
            raise ValueError("candidate must be an existing HIP shared library (.so)")
        self.path = path
        self.binary_sha256 = sha256_file(path)
        self.library = ctypes.CDLL(str(path.resolve()))
        try:
            self.function = getattr(self.library, ENTRYPOINT)
        except AttributeError as exc:
            raise ValueError(f"candidate has no {ENTRYPOINT} symbol") from exc
        self.function.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int32] * 6 + [
            ctypes.c_float, ctypes.c_void_p
        ]
        self.function.restype = ctypes.c_int

    def run(self, inputs: dict[str, torch.Tensor], out: torch.Tensor, scale: float) -> int:
        def ptr(key: str) -> ctypes.c_void_p:
            return ctypes.c_void_p(inputs[key].data_ptr())

        q = inputs["q"]
        status = self.function(
            ptr("q"), ptr("unified_kv"), ptr("kv_indices_prefix"),
            ptr("kv_indptr_prefix"), ptr("kv"), ptr("kv_indices_extend"),
            ptr("kv_indptr_extend"), ptr("attn_sink"), ctypes.c_void_p(out.data_ptr()),
            q.shape[0], q.shape[1], inputs["unified_kv"].shape[0], inputs["kv"].shape[0],
            inputs["kv_indices_prefix"].numel(), inputs["kv_indices_extend"].numel(),
            float(scale), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        return int(status)


def compare_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    guard_before: torch.Tensor,
    guard_after: torch.Tensor,
    inputs_unchanged: bool,
    return_code: int,
) -> dict:
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise ValueError("candidate output shape/dtype mismatch")
    finite = bool(torch.isfinite(actual).all())
    close = torch.isclose(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)
    mismatch_fraction = float((~close).float().mean())
    max_abs_error = float((actual.float() - expected.float()).abs().max())
    if not math.isfinite(max_abs_error):
        max_abs_error = None
    guards_ok = bool((guard_before == POISON).all() and (guard_after == POISON).all())
    nontrivial = not bool(expected.abs().any()) or bool(actual.abs().any())
    passed = (
        return_code == 0
        and finite
        and mismatch_fraction <= MAX_MISMATCH_FRACTION
        and guards_ok
        and inputs_unchanged
        and nontrivial
    )
    return {
        "passed": passed,
        "return_code": return_code,
        "finite": finite,
        "mismatch_fraction": mismatch_fraction,
        "max_abs_error": max_abs_error,
        "guards_ok": guards_ok,
        "inputs_unchanged": inputs_unchanged,
        "nontrivial": nontrivial,
    }


def run_case(candidate: HipCandidate, case: Case) -> dict:
    cpu_inputs = make_inputs(case)
    expected = oracle(cpu_inputs, case.softmax_scale)
    gpu_inputs = {key: tensor.cuda() for key, tensor in cpu_inputs.items()}
    storage = torch.full(
        (expected.numel() + 2 * GUARD,), POISON, dtype=torch.bfloat16, device="cuda"
    )
    out = storage[GUARD:-GUARD].view_as(expected)
    iterations = []
    for _ in range(2):
        out.fill_(POISON)
        return_code = candidate.run(gpu_inputs, out, case.softmax_scale)
        torch.cuda.synchronize()
        actual = out.cpu()
        inputs_unchanged = all(
            torch.equal(tensor.cpu(), cpu_inputs[key]) for key, tensor in gpu_inputs.items()
        )
        iterations.append(compare_output(
            actual, expected,
            guard_before=storage[:GUARD].cpu(),
            guard_after=storage[-GUARD:].cpu(),
            inputs_unchanged=inputs_unchanged,
            return_code=return_code,
        ))
        if not iterations[-1]["passed"]:
            break
    return {"name": case.name, "passed": len(iterations) == 2 and all(i["passed"] for i in iterations), "iterations": iterations}


def score(candidate_path: Path, aiter_source: Path, matrix: Path | None) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(aiter_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AITER_SHA:
        raise RuntimeError(f"AITER checkout is {revision}, expected {AITER_SHA}")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"expected gfx950, found {arch}")
    private = matrix is not None
    if private:
        require_outside_repo(matrix)
        if sha256_file(matrix) != PRIVATE_MATRIX_SHA256:
            raise RuntimeError("withheld matrix SHA256 does not match admission commitment")
    cases = load_cases(matrix)
    candidate = HipCandidate(candidate_path)
    findings = [run_case(candidate, case) for case in cases]
    passed_count = sum(item["passed"] for item in findings)
    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "aiter_sha": AITER_SHA,
        "arch": arch,
        "candidate_binary_sha256": candidate.binary_sha256,
        "case_count": len(findings),
        "passed_count": passed_count,
        "correctness": "withheld_pass" if private and passed_count == len(findings) else (
            "visible_pass" if not private and passed_count == len(findings) else "fail"
        ),
        "withheld_evaluated": private,
        "performance": "not_run",
        "scored_eligible": False,
    }
    if private:
        result["withheld_matrix_sha256_commitment"] = PRIVATE_MATRIX_SHA256
        result["failed_case_count"] = len(findings) - passed_count
    else:
        result["cases"] = findings
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--matrix", type=Path, help="withheld matrix; trusted runner only")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require_outside_repo(args.output_dir)
    if args.matrix:
        require_outside_repo(args.matrix)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        result = score(args.candidate, args.aiter_source, args.matrix)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "correctness": "setup_or_execution_error",
            "error": repr(exc),
            "performance": "not_run",
            "scored_eligible": False,
        }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result_file = args.output_dir / "result.json"
    descriptor = os.open(result_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps({
        "correctness": result["correctness"],
        "case_count": result.get("case_count"),
        "passed_count": result.get("passed_count"),
        "performance": "not_run",
    }))
    return 0 if result["correctness"] in ("visible_pass", "withheld_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
