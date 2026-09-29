"""Trusted, correctness-only admission probe for stable gfx950 decode top-k."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import secrets
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from references.topk_decode_gfx950 import Case, K, ROWS, WIDTH, make_logits, oracle, public_cases


def private_cases() -> tuple[Case, ...]:
    return (
        Case("random_full", "random", secrets.randbits(63), (WIDTH,) * ROWS),
        Case("heavy_ties", "ties", secrets.randbits(63), (WIDTH,) * ROWS),
        Case("short_padded", "padded", secrets.randbits(63), (127, 500, 1100, WIDTH)),
        Case("decode_next_n_2", "next_n", secrets.randbits(63), (49_913, WIDTH), next_n=2),
        Case("signed_zero", "signed_zero", secrets.randbits(63), (WIDTH,) * ROWS),
    )


def load_cases(path: Path | None) -> tuple[Case, ...]:
    if path is None:
        return public_cases()
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1 or raw.get("task_id") != "topk-decode-gfx950-stable-long":
        raise ValueError("unrecognized private matrix")
    if raw.get("aiter_sha") != "868ccf62a0bcad3aa47f92728340ccb37ed4fb39":
        raise ValueError("private matrix is pinned to a different AITER revision")
    if raw.get("generator_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("private matrix generator revision mismatch")
    cases = tuple(Case(**{**item, "seq_lens": tuple(item["seq_lens"])}) for item in raw["cases"])
    for case in cases:
        case.validate()
    return cases


def generate_private(path: Path) -> None:
    matrix = {
        "schema_version": 1,
        "task_id": "topk-decode-gfx950-stable-long",
        "aiter_sha": "868ccf62a0bcad3aa47f92728340ccb37ed4fb39",
        "generator": "references.topk_decode_admission.private_cases",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [asdict(case) for case in private_cases()],
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(matrix, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


def probe(aiter_source: Path, matrix: Path | None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"expected gfx950; found {arch}")
    sys.path.insert(0, str(aiter_source))
    topk = importlib.import_module("aiter.ops.topk")
    cases = load_cases(matrix)
    routes = {"flydsl_long": 0, "flydsl_radix_one_block": 0, "hip": 0}
    original = {
        "flydsl_long": topk.flydsl_top_k_per_row_decode,
        "flydsl_radix_one_block": topk.flydsl_radix_topk_one_block_decode,
        "hip": topk._hip_top_k_per_row_decode,
    }

    def counted(name):
        def call(*args, **kwargs):
            routes[name] += 1
            return original[name](*args, **kwargs)

        return call

    topk.flydsl_top_k_per_row_decode = counted("flydsl_long")
    topk.flydsl_radix_topk_one_block_decode = counted("flydsl_radix_one_block")
    topk._hip_top_k_per_row_decode = counted("hip")

    result = {"task_id": "topk-decode-gfx950-stable-long", "arch": arch, "cases": []}
    for case in cases:
        cpu_logits = make_logits(case)
        expected_indices, expected_values = oracle(case, cpu_logits)
        logits = cpu_logits.cuda()
        seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device="cuda")
        case_result = {"name": case.name, "passed": True, "iterations": []}
        for _ in range(2):
            indices = torch.full((case.rows, case.k), -777, dtype=torch.int32, device="cuda")
            values = torch.full((case.rows, case.k), float("nan"), dtype=torch.float32, device="cuda")
            before = routes.copy()
            topk.top_k_per_row_decode(
                logits, case.next_n, seq_lens, indices, case.rows,
                logits.stride(0), logits.stride(1), case.k, stable=True, values=values,
            )
            torch.cuda.synchronize()
            actual_indices = indices.cpu()
            actual_values = values.cpu()
            index_ok = torch.equal(actual_indices, expected_indices)
            value_bits_ok = torch.equal(actual_values.view(torch.int32), expected_values.view(torch.int32))
            used = {name: routes[name] - before[name] for name in routes}
            iteration = {
                "indices_equal": index_ok,
                "value_bits_equal": value_bits_ok,
                "route": used,
            }
            case_result["iterations"].append(iteration)
            case_result["passed"] &= index_ok and value_bits_ok and used == {
                "flydsl_long": 1, "flydsl_radix_one_block": 0, "hip": 0,
            }
        result["cases"].append(case_result)
    result["all_passed"] = all(case["passed"] for case in result["cases"])
    result["performance"] = "not_run"
    result["agent_trials"] = 0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-private", type=Path)
    parser.add_argument("--aiter-source", type=Path)
    parser.add_argument("--matrix", type=Path, help="private matrix, readable only by the trusted probe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.generate_private:
        generate_private(args.generate_private)
        return 0
    if not args.aiter_source or not args.output:
        parser.error("--aiter-source and --output are required for a probe")
    result = probe(args.aiter_source, args.matrix)
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
