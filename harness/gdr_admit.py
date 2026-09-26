"""Admit pinned AITER gfx950 GDR decode against an independent CPU oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from harness.core import merge_withheld, read_spec, sha256_path
from harness.run import _command, _gpu_manifest


def run_case(plugin, case: dict) -> dict:
    import torch

    plugin.validate_case(case)
    expected_state = plugin.make_initial_state(case)
    actual_state = plugin.guarded_state(
        expected_state, slot_padding=bool(case.get("state_slot_padding", False))
    )
    steps = []
    for step in range(case["steps"]):
        cpu_inputs = plugin.make_step_inputs(case, step)
        gpu_inputs, input_padding = plugin.gpu_step_inputs(cpu_inputs, case)
        output = plugin.guarded_output(case["batch"])
        expected_out = plugin.oracle_step(cpu_inputs, case["indices"], expected_state)
        returned_out, returned_state = plugin.run_aiter(gpu_inputs, actual_state.tensor, output.tensor)
        torch.cuda.synchronize()
        if returned_out.data_ptr() != output.tensor.data_ptr():
            raise AssertionError("AITER returned a different output allocation")
        if returned_state.data_ptr() != actual_state.tensor.data_ptr():
            raise AssertionError("AITER returned a different state allocation")
        plugin.compare_step(
            cpu_inputs, case["indices"], expected_out, expected_state,
            gpu_inputs, actual_state, output, input_padding,
        )
        steps.append({"step": step, "pass": True})
    return {"id": case["id"], "visibility": case["visibility"], "pass": True, "steps": steps}


def score(args, spec: dict) -> dict:
    revision = _command("git", "-C", str(args.aiter_source), "rev-parse", "HEAD").strip()
    if revision != spec["aiter_sha"]:
        raise RuntimeError(f"pinned AITER revision mismatch: {revision}")
    sys.path.insert(0, str(args.aiter_source))
    plugin = importlib.import_module(spec["plugin"])
    manifest = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)

    entries = []
    for case in spec["cases"]:
        try:
            entries.append(run_case(plugin, case))
        except Exception as exc:
            entries.append({
                "id": case["id"], "visibility": case["visibility"], "pass": False,
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
    all_pass = all(entry["pass"] for entry in entries)
    return {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "aiter_sha": spec["aiter_sha"],
        "public_spec_raw_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "public_spec_path_sha256": sha256_path(args.spec),
        "withheld_cases_sha256": spec["withheld_cases_sha256"],
        "withheld_cases_evaluated": args.withheld_spec is not None,
        "environment": manifest,
        "candidate_kind": "aiter_baseline_admission",
        "correctness": {
            "status": ("pass" if args.withheld_spec else "visible_pass") if all_pass else "fail",
            "case_count": len(entries), "cases": entries,
        },
        "performance": {"status": "not_run", "reason": "correctness_only"},
        "joint_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--withheld-spec", type=Path)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.aiter_source = args.aiter_source.resolve()
    args.host_gpu_report = args.host_gpu_report.resolve()
    args.output = args.output.resolve()
    spec = read_spec(args.spec)
    if args.withheld_spec:
        args.withheld_spec = args.withheld_spec.resolve()
        if args.output.is_relative_to(Path.cwd().resolve()):
            parser.error("withheld results must be outside the repository")
        spec = merge_withheld(spec, args.withheld_spec)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = score(args, spec)
    except Exception as exc:
        result = {
            "schema_version": 1, "task_id": spec["task_id"],
            "error": repr(exc), "traceback": traceback.format_exc(), "joint_pass": False,
        }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result_path = args.output / "result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result_path.chmod(0o444)
    status = result.get("correctness", {}).get("status")
    print(json.dumps({
        "result": str(result_path), "correctness": status,
        "case_count": result.get("correctness", {}).get("case_count"),
        "performance": "not_run", "error": result.get("error"),
    }))
    return 0 if status in ("pass", "visible_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
