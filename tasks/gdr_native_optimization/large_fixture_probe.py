"""Unscored batch-16 GDR graph feasibility for the pinned native-derived seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.core import read_spec, score_buckets
from harness.gdr_perf_probe import (
    MAX_LATENCY_RATIO,
    MAX_MAD_RATIO,
    REPEATS,
    WARMUP,
    benchmark_case,
)
from harness.gdr_score import run_case
from harness.run import _active_gpu_pids, _command, _gpu_manifest


AITERSHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
SPEC_SHA256 = "fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145"
FIXTURE_SHA256 = "10c53ca20475142a87cc515e917194ad07047bc54ef0dd8ca6e665038b023c0b"
SOURCE_SHA256 = "5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519"
BINARY_SHA256 = "8b66dd83364de6bbaed9c254e7d44a3183bc38445413e885ecfa86722870b709"
BUCKETS = ("valid_slots", "strided_mixed", "large_valid_slots_candidate")
GRAPH_CALLS = 32


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(path: Path, spec: dict) -> dict:
    if raw_sha256(path) != FIXTURE_SHA256:
        raise ValueError("large public fixture raw SHA256 mismatch")
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != 1 or fixture.get("task_id") != spec["task_id"]:
        raise ValueError("large public fixture has wrong task/schema")
    case = fixture.get("case")
    if not isinstance(case, dict) or case.get("visibility") != "visible":
        raise ValueError("large fixture must be one public case")
    if case.get("id") != BUCKETS[-1] or any(existing["id"] == case["id"] for existing in spec["cases"]):
        raise ValueError("large fixture ID is missing or collides with the original spec")
    if (case.get("batch"), case.get("pool"), case.get("steps"), case.get("state_slot_padding")) != (16, 20, 1, True):
        raise ValueError("large fixture shape/stride contract changed")
    from references.gdr_decode_packed_bf16 import validate_case

    validate_case(case)
    if len(case["indices"]) != 16 or any(not 0 <= index < 20 for index in case["indices"]):
        raise ValueError("large fixture requires unique valid state indices")
    return case


def quiet_gpu(*, require_self: bool) -> dict:
    report = _command("rocm-smi", "--showpids")
    if "KFD process information" not in report:
        raise RuntimeError("GPU PID table unavailable")
    pids = _active_gpu_pids(report)
    if require_self and os.getpid() not in pids:
        raise RuntimeError("scorer host PID not visible; use --pid=host")
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign:
        raise RuntimeError(f"foreign active GPU processes: {foreign}")
    return {"self_visible": os.getpid() in pids, "foreign_active_count": 0}


def summarize_improvement(performance: dict) -> None:
    ratios = [performance["buckets"][name]["ratio"] for name in BUCKETS]
    geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    performance["geomean_ratio"] = geomean
    performance["proposed_5pct_improvement_pass"] = (
        performance["pass"] and geomean <= 0.95
    )


def run_probe(args) -> dict:
    spec = read_spec(args.spec)
    if raw_sha256(args.spec) != SPEC_SHA256 or spec["aiter_sha"] != AITERSHA:
        raise ValueError("original public GDR spec changed")
    if raw_sha256(args.source) != SOURCE_SHA256 or raw_sha256(args.candidate) != BINARY_SHA256:
        raise ValueError("analyst source or binary changed")
    fixture = load_fixture(args.fixture, spec)
    revision = _command("git", "-C", str(args.aiter_source), "rev-parse", "HEAD").strip()
    if revision != AITERSHA:
        raise ValueError("pinned AITER source revision mismatch")
    preflight = quiet_gpu(require_self=False)

    sys.path.insert(0, str(args.aiter_source))
    plugin = importlib.import_module(spec["plugin"])
    candidate = plugin.load_hip_candidate(args.candidate)
    environment = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    cases = spec["cases"] + [fixture]
    correctness = [run_case(plugin, candidate, case) for case in cases]
    if not all(entry["pass"] for entry in correctness):
        return {
            "status": "correctness_failed", "correctness": correctness,
            "performance": {"status": "not_run"}, "environment": environment,
            "preflight": preflight,
        }

    by_id = {case["id"]: case for case in cases}
    samples = {
        name: benchmark_case(plugin, candidate, by_id[name], GRAPH_CALLS)
        for name in BUCKETS
    }
    postflight = quiet_gpu(require_self=True)
    performance = score_buckets(samples, MAX_LATENCY_RATIO, MAX_MAD_RATIO)
    summarize_improvement(performance)
    return {
        "status": "complete", "correctness": correctness,
        "performance": performance, "environment": environment,
        "preflight": preflight, "postflight": postflight,
        "post_graph_oracle_and_guards_pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for field in ("spec", "fixture", "source", "candidate", "aiter_source", "host_gpu_report", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if args.output.is_relative_to(Path.cwd().resolve()) or args.output.exists():
        parser.error("private output must be a new directory outside the repository")

    result = {
        "schema_version": 1,
        "kind": "unscored_gdr_large_fixture_feasibility",
        "scored_agent_candidate": False,
        "scored_eligible": False,
        "joint_pass": False,
        "withheld_cases_evaluated": False,
        "aiter_sha": AITERSHA,
        "spec_raw_sha256": SPEC_SHA256,
        "fixture_raw_sha256": FIXTURE_SHA256,
        "source_raw_sha256": SOURCE_SHA256,
        "candidate_binary_raw_sha256": BINARY_SHA256,
        "graph_calls": GRAPH_CALLS,
        "warmup": WARMUP,
        "paired_repeats": REPEATS,
        "bucket_case_ids": list(BUCKETS),
        "non_inferiority_ratio": MAX_LATENCY_RATIO,
        "max_mad_ratio": MAX_MAD_RATIO,
        "proposed_improvement_geomean_ratio": 0.95,
        "host_gap_caveat": "Graph replay amortizes host enqueue gaps; it is not a per-kernel profiler trace.",
    }
    try:
        result.update(run_probe(args))
    except Exception as exc:
        result.update({"status": "probe_error", "error": repr(exc), "performance": {"status": "not_run"}})
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "result.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output.chmod(0o444)
    print(json.dumps({
        "status": result["status"],
        "performance_pass": result.get("performance", {}).get("pass"),
        "result": str(output),
    }))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
