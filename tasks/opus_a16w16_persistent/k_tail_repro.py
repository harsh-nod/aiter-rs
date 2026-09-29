"""Unscored public analyst control for partial-K OPUS persistent behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tasks.opus_a16w16_persistent.admit import _guard_intact, _guarded_tensor, sha256_file
from tasks.opus_a16w16_persistent.adversarial_matrix import (
    compare_output,
    fp32_reference,
    make_operands,
    validate_case,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adversarial_admit as admission


FRESH_CASES = ((194, 95114), (254, 95115), (256, 95116))


def fresh_cases(base: dict) -> list[dict]:
    cases = []
    for k, seed in FRESH_CASES:
        case = {**base, "id": f"analyst_k{k}_fresh_random", "k": k, "seed": seed, "pattern": "random"}
        validate_case(case)
        cases.append(case)
    return cases


def padded_row_probe(case: dict) -> dict:
    import torch
    from aiter.ops.opus import opus_bmm

    m, n, k = case["m"], case["n"], case["k"]
    padded_k = 256
    logical_a, logical_b = make_operands(torch, case, device="cuda", phase=0)
    a_store, a_full, guard = _guarded_tensor(torch, (1, m, padded_k), sentinel=13.0)
    b_store, b_full, _ = _guarded_tensor(torch, (1, n, padded_k), sentinel=-17.0)
    y_store, y, _ = _guarded_tensor(torch, (1, m, n), sentinel=23.0)
    a = a_full[:, :, :k]
    b = b_full[:, :, :k]
    reference = fp32_reference(torch, logical_a, logical_b)
    bf16_reference = reference.to(torch.bfloat16).float()
    entries = []
    prior_output = None
    for pad_value in (0.0, 1.0):
        a_full.fill_(pad_value)
        b_full.fill_(pad_value)
        a.copy_(logical_a)
        b.copy_(logical_b)
        y.fill_(float("nan"))
        opus_bmm(a, b, y, kid=case["kid"], split_k=0)
        torch.cuda.synchronize()
        actual = y.float()
        check = compare_output(torch, y, reference)
        bf16_check = compare_output(torch, y, bf16_reference)
        entries.append({
            "row_padding_value": pad_value,
            "physical_stride_a": list(a.stride()),
            "physical_stride_b": list(b.stride()),
            "fp32_oracle": admission._clean_check(check),
            "bf16_cast_oracle": admission._clean_check(bf16_check),
            "input_unchanged": admission._bitwise_equal(torch, a, logical_a) and admission._bitwise_equal(torch, b, logical_b),
            "padding_unchanged": bool(torch.all(a_full[:, :, k:] == pad_value).item() and torch.all(b_full[:, :, k:] == pad_value).item()),
            "guards_intact": all((
                _guard_intact(torch, a_store, guard, 13.0),
                _guard_intact(torch, b_store, guard, -17.0),
                _guard_intact(torch, y_store, guard, 23.0),
            )),
            "output_differs_from_zero_pad": None if prior_output is None else not admission._bitwise_equal(torch, actual, prior_output),
        })
        if prior_output is None:
            prior_output = actual.clone()
    return {"padded_k": padded_k, "entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("matrix", "anchors", "aiter-source", "adapter-source", "adapter", "host-gpu-report", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    args = parser.parse_args()
    for field in ("matrix", "anchors", "aiter_source", "adapter_source", "adapter", "host_gpu_report", "output"):
        setattr(args, field, getattr(args, field).resolve())
    code_root = Path(__file__).resolve().parents[2]
    if args.output.exists() or args.output.is_relative_to(code_root) or args.output.is_relative_to(args.aiter_source):
        parser.error("output must be a new private off-source file")
    matrix = admission.check_provenance(args)
    base = admission.select_public_case(matrix, "k-partial-final-tile")
    if admission._gpu_pids():
        raise RuntimeError("foreign active GPU process at preflight")
    sys.path.insert(0, str(args.aiter_source))
    import torch

    environment = admission._gpu_attestation(torch, args.host_gpu_report.read_text(encoding="utf-8"))
    output = {
        "schema": "aiter-rs-opus-public-k-tail-analyst-repro-v1",
        "kind": "unscored_source_domain_investigation",
        "case_origin": "public K194 case with fresh public random seeds",
        "aiter_sha": admission.AITERSHA,
        "matrix_sha256": admission.MATRIX_SHA,
        "adapter_binary_sha256": admission.ADAPTER_BINARY_SHA,
        "host_gpu_report_sha256": sha256_file(args.host_gpu_report),
        "driver_sha256": sha256_file(Path(__file__)),
        "environment": environment,
        "withheld_cases_evaluated": False,
        "scored_agent_candidate": False,
        "performance": "not_run",
        "fresh_cases": [],
    }
    unsupported = {**base, "id": "analyst_k192_prelaunch", "k": 192, "pattern": "random"}
    try:
        validate_case(unsupported)
    except ValueError as exc:
        output["k192_prelaunch"] = {"rejected": True, "reason": str(exc)}
    else:
        output["k192_prelaunch"] = {"rejected": False}
    for case in fresh_cases(base):
        output["fresh_cases"].append(admission.run_case(case, args.adapter, args.host_gpu_report.read_text(encoding="utf-8")))
    try:
        output["k194_padded_row"] = padded_row_probe(fresh_cases(base)[0])
    except Exception as exc:
        output["k194_padded_row"] = {"completed": False, "error_type": type(exc).__name__, "error": str(exc)}
    active = admission._gpu_pids()
    output["host_pid_attributed"] = os.getpid() in active and not any(
        pid != os.getpid() for pid in active
    )
    output["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({
        "fresh_cases": [(case["case_id"], case["steps"][0]["baseline"]["bad_count"]) for case in output["fresh_cases"]],
        "padded": output["k194_padded_row"].get("completed", True),
    }))
    return 0 if output["host_pid_attributed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
