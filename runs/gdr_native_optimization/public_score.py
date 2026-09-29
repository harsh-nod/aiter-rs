"""Trusted public-only GDR correctness and graph feedback inside ROCm Docker.

No private manifest is accepted or mounted. Raw output stays in a private host
result directory; an outer broker may release only allowlisted feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

from runs.gdr_native_optimization.boundary import (
    compile_argv,
    sha256,
    validate_source_tree,
    validate_task,
)


class EnvironmentInvalidError(RuntimeError):
    pass


def verify_gpu_pid_probe(raw: str, active_pids: list[int], own_pid: int, stage: str) -> dict:
    if "KFD process information" not in raw or "PID" not in raw:
        raise EnvironmentInvalidError(f"{stage}: GPU process probe unavailable")
    if own_pid not in active_pids:
        raise EnvironmentInvalidError(f"{stage}: scorer PID not visible to ROCm")
    foreign = [pid for pid in active_pids if pid != own_pid]
    if foreign:
        raise EnvironmentInvalidError(f"{stage}: foreign active GPU processes present")
    return {"stage": stage, "self_pid_visible": True, "foreign_active_count": 0}


def default_stream_zero_case(plugin, candidate, case: dict) -> dict:
    import torch

    initial_state = plugin.make_initial_state(case)
    expected_state = initial_state.clone()
    cpu_inputs = plugin.make_step_inputs(case, 0)
    expected_out = plugin.oracle_step(cpu_inputs, case["indices"], expected_state)
    with torch.cuda.stream(torch.cuda.default_stream()):
        gpu_inputs, input_guards = plugin.gpu_step_inputs(cpu_inputs, case)
        gpu_state = plugin.guarded_state(
            initial_state, slot_padding=bool(case.get("state_slot_padding", False))
        )
        gpu_out = plugin.guarded_output(case["batch"])
        candidate.run(gpu_inputs, gpu_state.tensor, gpu_out.tensor, 0)
    torch.cuda.synchronize()
    plugin.compare_step(
        cpu_inputs, case["indices"], expected_out, expected_state,
        gpu_inputs, gpu_state, gpu_out, input_guards,
    )
    return {"pass": True, "case_id": case["id"], "hip_stream_handle": 0}


def run_public(args) -> dict:
    task, contract = validate_task(args.task_dir)
    if sha256(args.host_gpu_report) != task["host_gpu_report_sha256"]:
        raise RuntimeError("trusted host GPU report differs from task pin")
    repo = args.task_dir.parents[2]
    git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
    trusted_scopes = ["harness", "references"]
    if task["task_mode"] == "brokered_public_feedback":
        trusted_scopes += ["runs/runner.py", "runs/gdr_native_optimization"]
        if os.getuid() == 0 or args.output.stat().st_uid != os.getuid():
            raise RuntimeError("live scorer must run as the private output's non-root host UID")
        if pwd.getpwuid(os.getuid()).pw_name != "aiter-replay":
            raise RuntimeError("live scorer lacks its pinned minimal NSS identity")
        expected_env = {
            "HOME": "/tmp", "XDG_CACHE_HOME": "/tmp/.cache",
            "AITER_JIT_DIR": "/tmp/aiter-jit-cache",
        }
        if any(os.environ.get(name) != value for name, value in expected_env.items()):
            raise RuntimeError("live scorer cache/JIT environment differs from the task pin")
        Path(expected_env["AITER_JIT_DIR"]).mkdir(parents=True, exist_ok=True)
        if not os.access(expected_env["AITER_JIT_DIR"], os.W_OK):
            raise RuntimeError("live scorer JIT cache is not writable")
    subprocess.run(
        [*git, "diff", "--quiet", task["harness_revision"], "HEAD", "--", *trusted_scopes],
        check=True, timeout=30,
    )
    subprocess.run(
        [*git, "diff", "--quiet", "--", *trusted_scopes],
        check=True, timeout=30,
    )
    source_hashes = validate_source_tree(args.snapshot, {
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    })
    compiler = Path(shutil.which("hipcc") or "")
    if not compiler.is_file() or sha256(compiler) != task["compiler_hipcc_sha256"]:
        raise RuntimeError("hipcc binary differs from frozen compiler")
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={args.aiter_source}", "-C", str(args.aiter_source),
         "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != task["aiter_sha"]:
        raise RuntimeError("AITER checkout differs from frozen revision")
    subprocess.run(
        ["git", "-c", f"safe.directory={args.aiter_source}", "-C", str(args.aiter_source),
         "diff", "--quiet", "--"],
        check=True, timeout=30,
    )
    library = args.output / "libcandidate.so"
    command = compile_argv(args.snapshot, library, task)
    build = subprocess.run(command, cwd=args.output, capture_output=True, text=True,
                           timeout=180, check=False)
    raw = {
        "schema": "aiter-rs-gdr-opt-public-raw-v1",
        "source_sha256": source_hashes["kernel.hip"],
        "abi_header_sha256": task["abi_header_sha256"],
        "compiler_hipcc_sha256": task["compiler_hipcc_sha256"],
        "aiter_sha": revision,
        "public_spec_sha256": task["harness_spec_sha256"],
        "large_fixture_sha256": task["large_fixture_sha256"],
        "build_argv": ["hipcc", *task["build_flags"], "kernel.hip", "-o", "libcandidate.so"],
        "build_returncode": build.returncode,
    }
    if task["task_mode"] == "brokered_public_feedback":
        raw["container_execution"] = task["container_execution"]
        raw["container_uid"] = os.getuid()
    if build.returncode != 0 or not library.is_file():
        raw.update({
            "status": "compile_failed", "visible_case_results": {},
            "benchmark_bucket_results": {},
            "build_stderr_tail": build.stderr[-4096:],
        })
        return raw

    import torch
    from harness.core import read_spec, score_buckets
    from harness.gdr_perf_probe import benchmark_case
    from harness.gdr_score import run_case
    from harness.run import _active_gpu_pids, _command, _gpu_manifest

    if not torch.cuda.is_available():
        raise EnvironmentInvalidError("GPU unavailable")
    sys.path.insert(0, str(args.aiter_source))
    spec = read_spec(args.snapshot / "public/spec.json")
    fixture = json.loads((args.snapshot / "public/large_fixture.json").read_text())["case"]
    if [case["id"] for case in spec["cases"]] + [fixture["id"], "default_stream_zero"] != contract["visible_correctness_case_ids"]:
        raise ValueError("public cases differ from contract")
    try:
        environment = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    except RuntimeError as exc:
        raise EnvironmentInvalidError(str(exc)) from exc
    # Keep an allocated context alive so the scorer PID must be visible in rocm-smi.
    context_marker = torch.empty((1,), device="cuda")
    torch.cuda.synchronize()
    preflight_raw = _command("rocm-smi", "--showpids")
    preflight = verify_gpu_pid_probe(
        preflight_raw, _active_gpu_pids(preflight_raw), os.getpid(), "preflight",
    )
    plugin = importlib.import_module(spec["plugin"])
    candidate = plugin.load_hip_candidate(library)
    cases = spec["cases"] + [fixture]
    checked = [run_case(plugin, candidate, case) for case in cases]
    try:
        stream_zero = default_stream_zero_case(plugin, candidate, spec["cases"][0])
    except Exception as exc:
        stream_zero = {"pass": False, "case_id": spec["cases"][0]["id"],
                       "hip_stream_handle": 0, "error": repr(exc)}
    raw["arch"] = task["target_arch"]
    raw["gpu_name"] = environment["gpu_name"]
    raw["rocm_product"] = environment["rocm_product"]
    raw["host_gpu_report_sha256"] = environment["host_gpu_report_sha256"]
    raw["contention_preflight"] = preflight
    raw["candidate_binary_sha256"] = sha256(library)
    raw["correctness_details"] = checked
    raw["default_stream_zero"] = stream_zero
    raw["visible_case_results"] = {
        **{entry["id"]: bool(entry["pass"]) for entry in checked},
        "default_stream_zero": stream_zero["pass"],
    }
    if not all(entry["pass"] for entry in checked) or not stream_zero["pass"]:
        postflight_raw = _command("rocm-smi", "--showpids")
        raw["contention_postflight"] = verify_gpu_pid_probe(
            postflight_raw, _active_gpu_pids(postflight_raw), os.getpid(), "postflight"
        )
        raw.update({"status": "correctness_failed", "benchmark_bucket_results": {}})
        return raw
    if args.kind == "correctness":
        postflight_raw = _command("rocm-smi", "--showpids")
        raw["contention_postflight"] = verify_gpu_pid_probe(
            postflight_raw, _active_gpu_pids(postflight_raw), os.getpid(), "postflight"
        )
        raw.update({
            "status": "complete", "benchmark_bucket_results": {},
            "scored_eligible": False,
        })
        return raw

    selected = {case["id"]: case for case in cases}
    samples = {
        name: benchmark_case(plugin, candidate, selected[name], graph_repetitions=32)
        for name in contract["benchmark_case_ids"]
    }
    scored = score_buckets(
        samples,
        contract["benchmark_protocol"]["max_latency_ratio_each"],
        contract["benchmark_protocol"]["max_relative_mad_each_side"],
    )
    ratios = [scored["buckets"][name]["ratio"] for name in contract["benchmark_case_ids"]]
    postflight_raw = _command("rocm-smi", "--showpids")
    raw["contention_postflight"] = verify_gpu_pid_probe(
        postflight_raw, _active_gpu_pids(postflight_raw), os.getpid(), "postflight"
    )
    del context_marker
    raw.update({
        "status": "complete",
        "raw_performance": scored,
        "benchmark_bucket_results": {
            name: {"ratio": scored["buckets"][name]["ratio"],
                   "pass": scored["buckets"][name]["pass"]}
            for name in contract["benchmark_case_ids"]
        },
        "proposed_geomean_ratio": math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios)),
        "scored_eligible": False,
    })
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--kind", choices=("correctness", "benchmark"), required=True)
    args = parser.parse_args()
    for name in ("snapshot", "aiter_source", "task_dir", "output", "host_gpu_report"):
        setattr(args, name, getattr(args, name).resolve())
    repo = Path(__file__).resolve().parents[2]
    if args.output.is_relative_to(repo) or args.output.is_relative_to(args.snapshot):
        parser.error("raw public result must be outside code and snapshot")
    if not args.output.is_dir() or any(args.output.iterdir()):
        parser.error("raw public output must be an existing empty host-mounted directory")
    try:
        raw = run_public(args)
    except EnvironmentInvalidError as exc:
        raw = {
            "schema": "aiter-rs-gdr-opt-public-raw-v1",
            "status": "environment_invalid",
            "error": repr(exc),
            "visible_case_results": {}, "benchmark_bucket_results": {},
        }
    except Exception as exc:
        raw = {
            "schema": "aiter-rs-gdr-opt-public-raw-v1",
            "status": "setup_error",
            "error": repr(exc),
            "visible_case_results": {}, "benchmark_bucket_results": {},
        }
    path = args.output / "result.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(raw, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps({"status": raw["status"], "result_sha256": sha256(path)}))
    return 0 if raw["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
