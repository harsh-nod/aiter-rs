"""Unscored GDR HIP feasibility timing against pinned AITER on public cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.core import read_spec, score_buckets
from harness.gdr_score import run_case
from harness.run import _active_gpu_pids, _command, _gpu_manifest, _timed_call


BUCKET_CASE_IDS = ("valid_slots", "strided_mixed")
WARMUP = 5
REPEATS = 20
MAX_LATENCY_RATIO = 1.05
MAX_MAD_RATIO = 0.05


def benchmark_cases(spec: dict) -> list[dict]:
    cases = {case["id"]: case for case in spec["cases"]}
    missing = set(BUCKET_CASE_IDS) - cases.keys()
    if missing:
        raise ValueError(f"missing GDR feasibility cases: {sorted(missing)}")
    selected = [cases[case_id] for case_id in BUCKET_CASE_IDS]
    if any(case["steps"] != 1 or case["visibility"] != "visible" for case in selected):
        raise ValueError("GDR feasibility buckets must be visible one-step cases")
    return selected


def benchmark_case(plugin, candidate, case: dict, graph_repetitions: int = 1) -> dict:
    import torch

    cpu_state = plugin.make_initial_state(case)
    initial_state = cpu_state.to("cuda")
    cpu_inputs = plugin.make_step_inputs(case, 0)
    inputs, input_guards = plugin.gpu_step_inputs(cpu_inputs, case)
    aiter_state = plugin.guarded_state(
        cpu_state, slot_padding=bool(case.get("state_slot_padding", False))
    )
    hip_state = plugin.guarded_state(
        cpu_state, slot_padding=bool(case.get("state_slot_padding", False))
    )
    aiter_out = plugin.guarded_output(case["batch"])
    hip_out = plugin.guarded_output(case["batch"])

    def run_aiter() -> None:
        returned_out, returned_state = plugin.run_aiter(
            inputs, aiter_state.tensor, aiter_out.tensor
        )
        if returned_out.data_ptr() != aiter_out.tensor.data_ptr() or (
            returned_state.data_ptr() != aiter_state.tensor.data_ptr()
        ):
            raise AssertionError("AITER returned different allocations")

    def run_candidate() -> None:
        candidate.run(
            inputs, hip_state.tensor, hip_out.tensor,
            torch.cuda.current_stream().cuda_stream,
        )

    variants = {
        "aiter": (aiter_state, aiter_out, run_aiter),
        "candidate": (hip_state, hip_out, run_candidate),
    }

    def reset(name: str) -> None:
        state, output, _ = variants[name]
        state.tensor.copy_(initial_state)
        output.tensor.fill_(0)
        torch.cuda.synchronize()

    for _ in range(WARMUP):
        for name in variants:
            reset(name)
            variants[name][2]()
            torch.cuda.synchronize()

    timed_calls = {name: variant[2] for name, variant in variants.items()}
    if graph_repetitions > 1:
        graphs = {}
        for name, (_, _, call) in variants.items():
            reset(name)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_repetitions):
                    call()
            graphs[name] = graph
            timed_calls[name] = graph.replay
        for _ in range(WARMUP):
            for name in variants:
                reset(name)
                timed_calls[name]()
                torch.cuda.synchronize()

    samples = {"aiter_ms": [], "candidate_ms": []}
    for iteration in range(REPEATS):
        order = ("aiter", "candidate") if iteration % 2 == 0 else ("candidate", "aiter")
        for name in order:
            reset(name)
            elapsed = _timed_call(timed_calls[name]) / graph_repetitions
            samples["aiter_ms" if name == "aiter" else "candidate_ms"].append(elapsed)

    expected_state = cpu_state.clone()
    for _ in range(graph_repetitions):
        expected_out = plugin.oracle_step(cpu_inputs, case["indices"], expected_state)
    for state, output, _ in variants.values():
        plugin.compare_step(
            cpu_inputs, case["indices"], expected_out, expected_state,
            inputs, state, output, input_guards,
        )
    return samples


def run_probe(args) -> dict:
    spec = read_spec(args.spec)
    if hashlib.sha256(args.spec.read_bytes()).hexdigest() != args.spec_sha256:
        raise ValueError("public GDR spec differs from preregistered raw SHA256")
    if hashlib.sha256(args.candidate.read_bytes()).hexdigest() != args.candidate_sha256:
        raise ValueError("HIP candidate differs from preregistered binary SHA256")
    revision = _command("git", "-C", str(args.aiter_source), "rev-parse", "HEAD").strip()
    if revision != spec["aiter_sha"]:
        raise ValueError(f"pinned AITER revision mismatch: {revision}")
    selected = benchmark_cases(spec)

    preflight = _command("rocm-smi", "--showpids")
    if "KFD process information" not in preflight or _active_gpu_pids(preflight):
        raise RuntimeError("GPU process preflight unavailable or device occupied")

    sys.path.insert(0, str(args.aiter_source))
    plugin = importlib.import_module(spec["plugin"])
    candidate = plugin.load_hip_candidate(args.candidate)
    environment = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    correctness = [run_case(plugin, candidate, case) for case in spec["cases"]]
    if not all(entry["pass"] for entry in correctness):
        return {
            "status": "correctness_failed", "correctness": correctness,
            "performance": {"status": "not_run"}, "environment": environment,
        }

    samples = {
        case["id"]: benchmark_case(plugin, candidate, case, args.graph_repetitions)
        for case in selected
    }
    postflight = _command("rocm-smi", "--showpids")
    if "KFD process information" not in postflight:
        raise RuntimeError("GPU process postflight unavailable")
    foreign_pids = [pid for pid in _active_gpu_pids(postflight) if pid != os.getpid()]
    if foreign_pids:
        raise RuntimeError(f"foreign GPU processes observed after timing: {foreign_pids}")

    return {
        "status": "complete", "correctness": correctness,
        "performance": score_buckets(samples, MAX_LATENCY_RATIO, MAX_MAD_RATIO),
        "environment": environment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graph-repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.graph_repetitions < 1:
        parser.error("--graph-repetitions must be positive")
    for field in ("spec", "candidate", "aiter_source", "host_gpu_report", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if args.output.is_relative_to(Path.cwd().resolve()):
        parser.error("raw GDR timing result must be outside the repository")
    if args.output.exists():
        parser.error("output path already exists")

    result = {
        "schema_version": 1, "kind": "unscored_gdr_feasibility_probe",
        "scored_agent_candidate": False, "joint_pass": False,
        "aiter_sha": "868ccf62a0bcad3aa47f92728340ccb37ed4fb39",
        "spec_raw_sha256": args.spec_sha256,
        "candidate_binary_raw_sha256": args.candidate_sha256,
        "warmup": WARMUP, "repeats": REPEATS,
        "graph_repetitions": args.graph_repetitions,
        "timing_mode": (
            "cuda_graph_replay" if args.graph_repetitions > 1 else "direct_call_gpu_events"
        ),
        "bucket_case_ids": list(BUCKET_CASE_IDS),
        "threshold_ratio": MAX_LATENCY_RATIO, "max_mad_ratio": MAX_MAD_RATIO,
    }
    try:
        result.update(run_probe(args))
    except Exception as exc:
        result.update({"status": "probe_error", "error": repr(exc)})
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True)
    output_file = args.output / "result.json"
    output_file.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    output_file.chmod(0o444)
    print(json.dumps({
        "status": result["status"],
        "performance_pass": result.get("performance", {}).get("pass"),
        "result": str(output_file),
    }))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
