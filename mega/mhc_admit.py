"""Admit pinned gfx950 mHC post/pre against an independent four-output CPU oracle."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from harness.run import _gpu_manifest
from mega.mhc_oracle import Parameters, make_inputs, post_pre, validate_case


RECORDED_CALLS = (
    "mhc_fused_post_pre_gemm_sqrsum",
    "mhc_pre_gemm_sqrsum",
    "mhc_pre_big_fuse",
    "mhc_pre_big_fuse_rmsnorm",
    "mhc_fused_post_pre_large_m",
    "mhc_post",
    "mhc_pre",
)
OUTPUT_NAMES = ("post_mix", "comb_mix", "layer_input_out", "next_residual")


def expected_dispatch(case: dict) -> set[str]:
    if case["route"] == "large_m_fallback":
        return {"mhc_fused_post_pre_large_m", "mhc_post", "mhc_pre", "mhc_pre_gemm_sqrsum", "mhc_pre_big_fuse"}
    reduction = "mhc_pre_big_fuse_rmsnorm" if case.get("norm") else "mhc_pre_big_fuse"
    return {"mhc_fused_post_pre_gemm_sqrsum", reduction}


@contextlib.contextmanager
def record_dispatch(module):
    calls = []
    with contextlib.ExitStack() as stack:
        for name in RECORDED_CALLS:
            original = getattr(module, name)

            @functools.wraps(original)
            def wrapper(*args, _original=original, _name=name, **kwargs):
                calls.append(_name)
                return _original(*args, **kwargs)

            stack.enter_context(mock.patch.object(module, name, wrapper))
        yield calls


def compare_output(expected, actual, name: str) -> dict:
    import torch

    result = {
        "shape": list(actual.shape), "dtype": str(actual.dtype),
        "expected_shape": list(expected.shape), "expected_dtype": str(expected.dtype),
    }
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {**result, "pass": False, "reason": "shape_or_dtype"}
    observed, wanted = actual.float(), expected.float()
    if not bool(torch.isfinite(observed).all()):
        return {**result, "pass": False, "reason": "nonfinite"}
    delta = (observed - wanted).abs()
    mismatches = ~torch.isclose(observed, wanted, rtol=1e-2, atol=1e-2)
    count = int(mismatches.sum().item())
    return {
        **result, "pass": count == 0, "mismatch_count": count,
        "element_count": expected.numel(), "max_abs_error": float(delta.max().item()),
    }


def run_case(module, case: dict) -> dict:
    import torch

    validate_case(case)
    params = Parameters()
    inputs_cpu = make_inputs(case)
    expected = post_pre(inputs_cpu, params)
    inputs_gpu = {name: value.to("cuda") for name, value in inputs_cpu.items()}
    config = (
        module.get_mhc_fused_post_pre_config(case["m"], case["hidden_size"], False, False)
        if case["route"] == "fused" else None
    )
    kwargs = {
        "rms_eps": params.rms_eps,
        "hc_pre_eps": params.pre_eps,
        "hc_sinkhorn_eps": params.sinkhorn_eps,
        "hc_post_mult_value": params.post_multiplier,
        "sinkhorn_repeat": params.sinkhorn_repeat,
        "norm_weight": inputs_gpu.get("norm_weight"),
        "norm_eps": params.norm_eps,
        "force_fused": True,
        "w_preshuffle_bf16": False,
        "res_preshuffle": False,
    }
    with record_dispatch(module) as calls:
        actual = module.mhc_fused_post_pre(
            inputs_gpu["layer_input"], inputs_gpu["residual_in"],
            inputs_gpu["post_layer_mix"], inputs_gpu["comb_res_mix"],
            inputs_gpu["fn"], inputs_gpu["hc_scale"], inputs_gpu["hc_base"],
            **kwargs,
        )
        torch.cuda.synchronize()
    actual_cpu = tuple(value.cpu() for value in actual)
    outputs = {
        name: compare_output(wanted, observed, name)
        for name, wanted, observed in zip(OUTPUT_NAMES, expected, actual_cpu, strict=True)
    }
    inputs_unchanged = all(
        torch.equal(inputs_gpu[name].cpu().contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))
        for name, value in inputs_cpu.items()
    )
    required = expected_dispatch(case)
    dispatch_ok = required.issubset(calls) and all(
        name not in calls for name in (set(RECORDED_CALLS) - required)
    )
    return {
        "id": case["id"], "route": case["route"], "m": case["m"],
        "hidden_size": case["hidden_size"], "norm": bool(case.get("norm")),
        "config": list(config) if config is not None else None,
        "dispatch_calls": calls, "dispatch_pass": dispatch_ok,
        "inputs_unchanged": inputs_unchanged, "outputs": outputs,
        "pass": dispatch_ok and inputs_unchanged and all(item["pass"] for item in outputs.values()),
    }


def admit(args) -> dict:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    revision = subprocess.check_output(
        ["git", "-C", str(args.aiter_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != spec["aiter_sha"]:
        raise RuntimeError(f"pinned AITER revision mismatch: {revision}")
    sys.path.insert(0, str(args.aiter_source))
    module = importlib.import_module("aiter.ops.mhc")
    manifest = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    arch = module.get_gfx_runtime()
    cu_count = module.get_cu_num()
    if arch != "gfx950":
        raise RuntimeError(f"runtime MHC arch is {arch}, expected gfx950")
    if cu_count != 256:
        raise RuntimeError(f"expected gfx950 tuned 256-CU policy, got {cu_count} CUs")

    selected_ids = set(args.case or [case["id"] for case in spec["cases"]])
    known_ids = {case["id"] for case in spec["cases"]}
    if selected_ids - known_ids:
        raise ValueError(f"unknown cases: {sorted(selected_ids - known_ids)}")
    entries = []
    for case in spec["cases"]:
        if case["id"] not in selected_ids:
            continue
        try:
            entries.append(run_case(module, case))
        except Exception as exc:
            entries.append({
                "id": case["id"], "route": case["route"], "pass": False,
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
    complete = len(entries) == len(spec["cases"])
    all_pass = all(item["pass"] for item in entries)
    return {
        "schema_version": 1, "task_id": spec["task_id"],
        "aiter_sha": revision,
        "spec_raw_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "environment": manifest, "runtime_arch": arch, "runtime_cu_count": cu_count,
        "force_fused": True, "weight_mode": "fp32", "residual_layout": "plain",
        "correctness": {
            "status": ("complete_pass" if complete else "partial_pass") if all_pass else "fail",
            "complete_matrix": complete, "case_count": len(entries), "cases": entries,
        },
        "performance": {"status": "not_run", "reason": "admission_correctness_only"},
        "scored_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", help="bounded case subset; repeat as needed")
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.aiter_source = args.aiter_source.resolve()
    args.host_gpu_report = args.host_gpu_report.resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(Path.cwd().resolve()):
        parser.error("raw admission result must be outside the repository")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = admit(args)
    except Exception as exc:
        result = {
            "schema_version": 1, "error": repr(exc), "traceback": traceback.format_exc(),
            "correctness": {"status": "setup_error"},
            "performance": {"status": "not_run"}, "scored_eligible": False,
        }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result_path = args.output / "result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    result_path.chmod(0o444)
    status = result["correctness"]["status"]
    print(json.dumps({"correctness": status, "case_count": result["correctness"].get("case_count"), "performance": "not_run", "result": str(result_path)}))
    return 0 if status in ("partial_pass", "complete_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
