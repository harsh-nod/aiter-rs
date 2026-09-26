"""Correctness-only gfx950 GDR HIP candidate pilot, separate from AITER admission."""

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


def run_implementation(plugin, case: dict, invoke) -> dict:
    import torch

    expected_state = plugin.make_initial_state(case)
    actual_state = plugin.guarded_state(
        expected_state, slot_padding=bool(case.get("state_slot_padding", False))
    )
    steps = []
    for step in range(case["steps"]):
        try:
            cpu_inputs = plugin.make_step_inputs(case, step)
            gpu_inputs, input_guards = plugin.gpu_step_inputs(cpu_inputs, case)
            output = plugin.guarded_output(case["batch"])
            expected_out = plugin.oracle_step(cpu_inputs, case["indices"], expected_state)
            invoke(gpu_inputs, actual_state.tensor, output.tensor)
            torch.cuda.synchronize()
            plugin.compare_step(
                cpu_inputs, case["indices"], expected_out, expected_state,
                gpu_inputs, actual_state, output, input_guards,
            )
            steps.append({"step": step, "pass": True})
        except Exception as exc:
            steps.append({
                "step": step, "pass": False, "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            break
    return {"pass": len(steps) == case["steps"] and all(item["pass"] for item in steps), "steps": steps}


def run_case(plugin, candidate, case: dict) -> dict:
    import torch

    plugin.validate_case(case)

    def invoke_aiter(inputs, state, out):
        returned_out, returned_state = plugin.run_aiter(inputs, state, out)
        if returned_out.data_ptr() != out.data_ptr():
            raise AssertionError("AITER returned a different output allocation")
        if returned_state.data_ptr() != state.data_ptr():
            raise AssertionError("AITER returned a different state allocation")

    def invoke_candidate(inputs, state, out):
        candidate.run(inputs, state, out, torch.cuda.current_stream().cuda_stream)

    aiter = run_implementation(plugin, case, invoke_aiter)
    proposed = run_implementation(plugin, case, invoke_candidate)
    return {
        "id": case["id"], "visibility": case["visibility"],
        "aiter": aiter, "candidate": proposed,
        "pass": aiter["pass"] and proposed["pass"],
    }


def correctness_status(entries: list[dict], withheld: bool) -> str:
    if not all(entry.get("aiter", {}).get("pass") is True for entry in entries):
        return "baseline_invalid"
    if not all(entry.get("candidate", {}).get("pass") is True for entry in entries):
        return "fail"
    return "pass" if withheld else "visible_pass"


def score(args, spec: dict) -> dict:
    revision = _command("git", "-C", str(args.aiter_source), "rev-parse", "HEAD").strip()
    if revision != spec["aiter_sha"]:
        raise RuntimeError(f"pinned AITER revision mismatch: {revision}")
    sys.path.insert(0, str(args.aiter_source))
    plugin = importlib.import_module(spec["plugin"])
    manifest = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    candidate = plugin.load_hip_candidate(args.candidate)

    entries = []
    for case in spec["cases"]:
        try:
            entries.append(run_case(plugin, candidate, case))
        except Exception as exc:
            entries.append({
                "id": case["id"], "visibility": case["visibility"], "pass": False,
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
    return {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "aiter_sha": spec["aiter_sha"],
        "public_spec_raw_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "public_spec_path_sha256": sha256_path(args.spec),
        "candidate_binary_raw_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "candidate_path_sha256": sha256_path(args.candidate),
        "withheld_cases_sha256": spec["withheld_cases_sha256"],
        "withheld_cases_evaluated": args.withheld_spec is not None,
        "environment": manifest,
        "candidate_kind": "hip_candidate_correctness_pilot",
        "scored_agent_candidate": False,
        "correctness": {
            "status": correctness_status(entries, args.withheld_spec is not None),
            "case_count": len(entries), "cases": entries,
        },
        "performance": {"status": "not_run", "reason": "pilot_correctness_only"},
        "joint_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--withheld-spec", type=Path)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.candidate = args.candidate.resolve()
    args.aiter_source = args.aiter_source.resolve()
    args.host_gpu_report = args.host_gpu_report.resolve()
    args.output = args.output.resolve()
    if args.candidate.suffix != ".so" or not args.candidate.is_file():
        parser.error("candidate must be an existing HIP shared library (.so)")
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
            "error": repr(exc), "traceback": traceback.format_exc(),
            "correctness": {"status": "setup_error"},
            "performance": {"status": "not_run", "reason": "setup_error"},
            "joint_pass": False,
        }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result_path = args.output / "result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result_path.chmod(0o444)
    status = result["correctness"]["status"]
    print(json.dumps({
        "result": str(result_path), "correctness": status,
        "case_count": result["correctness"].get("case_count"),
        "performance": "not_run",
    }))
    return 0 if status in ("pass", "visible_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
