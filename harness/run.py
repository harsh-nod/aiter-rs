"""Score a candidate against a pinned AITER operator and independent oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from harness.core import merge_withheld, read_spec, score_buckets, sha256_path


def _command(*args: str) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return (result.stdout + result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def _active_gpu_pids(rocm_smi_output: str) -> list[int]:
    active = []
    for line in rocm_smi_output.splitlines():
        match = re.match(r"^\s*(\d+)\s+\S+\s+\S+\s+(\d+)\s+\d+\s+(\d+)\s*$", line)
        if match and (int(match.group(2)) > 0 or int(match.group(3)) > 0):
            active.append(int(match.group(1)))
    return sorted(set(active))


def _gpu_matches(spec: dict, torch_name: str, arch: str, rocm_product: str, host_report: dict | None) -> bool:
    if spec["target_arch"] not in arch or spec["gpu_pci_device_id"] not in rocm_product:
        return False
    if spec["gpu_sku"] in (torch_name + "\n" + rocm_product):
        return True
    return bool(
        host_report
        and host_report.get("gpu_name") == spec["gpu_sku"]
        and host_report.get("card_model") == spec["gpu_pci_device_id"]
        and host_report.get("arch") == spec["target_arch"]
        and spec["gpu_pci_device_id"] in rocm_product
    )


def _mismatches(actual: bytes, expected: bytes) -> dict:
    if len(actual) != len(expected):
        return {"pass": False, "expected_bytes": len(expected), "actual_bytes": len(actual)}
    positions = [index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]]
    return {"pass": not positions, "mismatch_count": len(positions), "first_mismatch_offsets": positions[:8]}


def _check_case(plugin, candidate, case: dict) -> dict:
    import torch

    x = plugin.make_input(case)
    original = x.clone()
    oracle_packed, oracle_scale = plugin.oracle(x)
    entry = {"id": case["id"], "bucket": case["bucket"]}

    baseline = plugin.run_aiter(x)
    torch.cuda.synchronize()
    plugin.validate_output(baseline, x)
    entry["aiter"] = {
        "packed": _mismatches(plugin.output_bytes(baseline)[0], oracle_packed),
        "scale": _mismatches(plugin.output_bytes(baseline)[1], oracle_scale),
        "input_unchanged": bool(torch.equal(x.view(torch.int16), original.view(torch.int16))),
    }
    entry["aiter"]["pass"] = all(
        item["pass"] for item in (entry["aiter"]["packed"], entry["aiter"]["scale"])
    ) and entry["aiter"]["input_unchanged"]
    if not entry["aiter"]["pass"]:
        entry["candidate"] = {"pass": None, "reason": "baseline_oracle_disagreement"}
        return entry

    candidate_output = candidate.run(x)
    torch.cuda.synchronize()
    plugin.validate_output(candidate_output, x, baseline)
    got_packed, got_scale = plugin.output_bytes(candidate_output)
    entry["candidate"] = {
        "packed": _mismatches(got_packed, oracle_packed),
        "scale": _mismatches(got_scale, oracle_scale),
        "input_unchanged": bool(torch.equal(x.view(torch.int16), original.view(torch.int16))),
    }
    entry["candidate"]["pass"] = all(
        item["pass"] for item in (entry["candidate"]["packed"], entry["candidate"]["scale"])
    ) and entry["candidate"]["input_unchanged"]
    return entry


def _timed_call(fn) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _benchmark_case(plugin, candidate, case: dict, warmup: int, repeats: int) -> dict:
    import torch

    x = plugin.make_input(case)
    baseline_fn, candidate_fn, check_guards = plugin.benchmark_calls(x, candidate)
    for _ in range(warmup):
        baseline_fn()
        candidate_fn()
    torch.cuda.synchronize()
    aiter, proposed = [], []
    for iteration in range(repeats):
        order = (("aiter", baseline_fn), ("candidate", candidate_fn))
        if iteration % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            (aiter if name == "aiter" else proposed).append(_timed_call(fn))
    torch.cuda.synchronize()
    check_guards()
    return {"aiter_ms": aiter, "candidate_ms": proposed}


def _gpu_manifest(spec: dict, aiter_source: Path, host_report_path: Path | None) -> dict:
    import aiter
    import torch

    if not Path(aiter.__file__).resolve().is_relative_to(aiter_source.resolve()):
        raise RuntimeError(f"AITER imported from unpinned path: {aiter.__file__}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the scorer")
    props = torch.cuda.get_device_properties(0)
    name = props.name
    arch = getattr(props, "gcnArchName", "")
    product = _command("rocm-smi", "--showproductname")
    host_report = json.loads(host_report_path.read_text()) if host_report_path else None
    if not _gpu_matches(spec, name, arch, product, host_report):
        raise RuntimeError(f"GPU mismatch: torch name={name!r}, ROCm product={product!r}, arch={arch!r}")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "aiter_package": str(Path(aiter.__file__).resolve()),
        "gpu_name": name,
        "gpu_arch": arch,
        "hipcc_version": _command("hipcc", "--version"),
        "rocm_product": product,
        "host_gpu_report_sha256": hashlib.sha256(host_report_path.read_bytes()).hexdigest() if host_report_path else None,
        "host_gpu_name": host_report["gpu_name"] if host_report else None,
        "host_card_model": host_report["card_model"] if host_report else None,
        "rocm_clocks": _command("rocm-smi", "--showclocks"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _run(args, spec: dict, candidate_path: Path, source: Path) -> dict:
    result = {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "aiter_sha": spec["aiter_sha"],
        "spec_sha256": sha256_path(args.spec),
        "candidate_path_sha256": sha256_path(candidate_path),
        "task_freeze_sha256": args.task_freeze_sha256,
        "final_tree_sha256": args.final_tree_sha256,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "correctness": {"status": "not_run"},
        "performance": {"status": "not_run"},
        "joint_pass": False,
        "scored_agent_candidate": not args.unscored_reference,
        "withheld_cases_sha256": spec["withheld_cases_sha256"],
        "withheld_cases_evaluated": args.withheld_spec is not None,
    }
    revision = _command("git", "-C", str(source), "rev-parse", "HEAD").strip()
    if revision != spec["aiter_sha"]:
        raise RuntimeError(f"AITER source revision mismatch: {revision}")

    before_text = _command("rocm-smi", "--showpids")
    before_pids = _active_gpu_pids(before_text)
    process_probe_valid = "KFD process information" in before_text and "PID" in before_text
    result["contention_preflight"] = {"active_pids": before_pids, "valid": process_probe_valid, "raw": before_text}

    sys.path.insert(0, str(source))
    plugin = importlib.import_module(spec["plugin"])
    candidate = plugin.load_hip_candidate(candidate_path)
    result["environment"] = _gpu_manifest(spec, source, args.host_gpu_report)

    cases = []
    for case in spec["cases"]:
        try:
            cases.append(_check_case(plugin, candidate, case))
        except Exception as exc:  # keep all attempted cases in the artifact
            cases.append({"id": case["id"], "bucket": case["bucket"], "error": repr(exc), "traceback": traceback.format_exc()})
    baseline_valid = all(item.get("aiter", {}).get("pass") is True for item in cases)
    candidate_correct = baseline_valid and all(item.get("candidate", {}).get("pass") is True for item in cases)
    result["correctness"] = {
        "status": (
            ("pass" if args.withheld_spec else "visible_pass")
            if candidate_correct else ("baseline_invalid" if not baseline_valid else "fail")
        ),
        "cases": cases,
    }
    if not candidate_correct:
        result["performance"] = {"status": "not_run", "reason": "correctness_not_passed"}
        return result
    if args.correctness_only:
        result["performance"] = {"status": "not_run", "reason": "correctness_only"}
        return result
    if not process_probe_valid:
        result["performance"] = {"status": "invalid", "reason": "process_probe_unavailable"}
        return result
    if before_pids:
        result["performance"] = {"status": "invalid", "reason": "preexisting_gpu_process", "active_pids": before_pids}
        return result

    selected = set(spec["benchmark_case_ids"])
    timings = {}
    for case in spec["cases"]:
        if case["id"] in selected:
            bucket = case["bucket"]
            if bucket in timings:
                raise ValueError(f"multiple benchmark cases share bucket {bucket}; define a separate bucket")
            timings[bucket] = _benchmark_case(plugin, candidate, case, int(spec["warmup"]), int(spec["repeats"]))
    after_text = _command("rocm-smi", "--showpids")
    if "KFD process information" not in after_text or "PID" not in after_text:
        result["performance"] = {"status": "invalid", "reason": "postflight_process_probe_unavailable", "samples": timings}
        return result
    all_after_pids = _active_gpu_pids(after_text)
    if os.getpid() not in all_after_pids:
        result["performance"] = {"status": "invalid", "reason": "scorer_host_pid_not_visible; use Docker --pid=host", "samples": timings}
        return result
    after_pids = [pid for pid in all_after_pids if pid != os.getpid()]
    result["contention_postflight"] = {"active_other_pids": after_pids, "raw": after_text}
    if after_pids:
        result["performance"] = {"status": "invalid", "reason": "gpu_process_detected_during_run", "samples": timings}
        return result
    scoring = score_buckets(timings, float(spec["threshold_ratio"]), float(spec["max_mad_ratio"]))
    result["performance"] = {"status": "pass" if scoring["pass"] else "fail", **scoring}
    result["joint_pass"] = scoring["pass"] and result["scored_agent_candidate"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", type=Path, help="trusted host-side ROCm GPU report")
    parser.add_argument("--withheld-spec", type=Path, help="private runner-only case manifest")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--unscored-reference", action="store_true", help="do not count a HIP reference as an agent attempt")
    parser.add_argument("--task-freeze-sha256")
    parser.add_argument("--final-tree-sha256")
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.candidate = args.candidate.resolve()
    args.aiter_source = args.aiter_source.resolve()
    if args.host_gpu_report:
        args.host_gpu_report = args.host_gpu_report.resolve()
    spec = read_spec(args.spec)
    if args.withheld_spec:
        args.withheld_spec = args.withheld_spec.resolve()
        spec = merge_withheld(spec, args.withheld_spec)
    elif not args.unscored_reference:
        parser.error("scored attempts require --withheld-spec")
    if args.candidate.suffix != ".so":
        parser.error("scored candidate must be a HIP shared library (.so)")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(args, spec, args.candidate, args.aiter_source)
    except Exception as exc:
        result = {"schema_version": 1, "task_id": spec["task_id"], "error": repr(exc), "traceback": traceback.format_exc(), "joint_pass": False}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result_path = args.output / "result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result_path.chmod(0o444)
    print(json.dumps({"result": str(result_path), "correctness": result.get("correctness", {}).get("status"), "performance": result.get("performance", {}).get("status"), "joint_pass": result["joint_pass"], "error": result.get("error")}))
    return 0 if result["joint_pass"] or (
        args.correctness_only and result.get("correctness", {}).get("status") in ("pass", "visible_pass")
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
