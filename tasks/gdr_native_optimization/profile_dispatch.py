"""One-call, correctness-checked GDR dispatch trace driver for rocprofv3."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.core import read_spec
from harness.run import _active_gpu_pids, _command, _gpu_manifest
from tasks.gdr_native_optimization.large_fixture_probe import (
    AITERSHA,
    BINARY_SHA256,
    BUCKETS,
    SOURCE_SHA256,
    SPEC_SHA256,
    load_fixture,
    raw_sha256,
)


def select_case(spec: dict, fixture: dict, case_id: str) -> dict:
    cases = {case["id"]: case for case in spec["cases"] + [fixture]}
    if case_id not in BUCKETS or case_id not in cases:
        raise ValueError("profile case is not an admitted public bucket")
    return cases[case_id]


def one_call(plugin, candidate, case: dict, mode: str) -> None:
    import torch

    cpu_state = plugin.make_initial_state(case)
    expected_state = cpu_state.clone()
    cpu_inputs = plugin.make_step_inputs(case, 0)
    inputs, input_guards = plugin.gpu_step_inputs(cpu_inputs, case)
    state = plugin.guarded_state(cpu_state, slot_padding=bool(case.get("state_slot_padding", False)))
    output = plugin.guarded_output(case["batch"])
    expected_out = plugin.oracle_step(cpu_inputs, case["indices"], expected_state)
    if mode == "aiter":
        returned_out, returned_state = plugin.run_aiter(inputs, state.tensor, output.tensor)
        if returned_out.data_ptr() != output.tensor.data_ptr() or returned_state.data_ptr() != state.tensor.data_ptr():
            raise AssertionError("AITER returned different allocations")
    else:
        candidate.run(inputs, state.tensor, output.tensor, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    plugin.compare_step(
        cpu_inputs, case["indices"], expected_out, expected_state,
        inputs, state, output, input_guards,
    )


def run(args) -> dict:
    spec = read_spec(args.spec)
    if raw_sha256(args.spec) != SPEC_SHA256 or spec["aiter_sha"] != AITERSHA:
        raise ValueError("original public GDR spec changed")
    if raw_sha256(args.source) != SOURCE_SHA256 or raw_sha256(args.candidate) != BINARY_SHA256:
        raise ValueError("native-derived source or binary changed")
    fixture = load_fixture(args.fixture, spec)
    case = select_case(spec, fixture, args.case)
    revision = subprocess.check_output(
        ["git", "-C", str(args.aiter_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AITERSHA:
        raise ValueError("pinned AITER source changed")
    preflight = _command("rocm-smi", "--showpids")
    if "KFD process information" not in preflight or _active_gpu_pids(preflight):
        raise RuntimeError("GPU process preflight unavailable or device occupied")

    sys.path.insert(0, str(args.aiter_source))
    plugin = importlib.import_module(spec["plugin"])
    environment = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    candidate = plugin.load_hip_candidate(args.candidate) if args.mode == "hip" else None
    one_call(plugin, candidate, case, args.mode)
    postflight = _command("rocm-smi", "--showpids")
    active = _active_gpu_pids(postflight)
    if "KFD process information" not in postflight or os.getpid() not in active:
        raise RuntimeError("host PID attribution unavailable; run with --pid=host")
    if any(pid != os.getpid() for pid in active):
        raise RuntimeError("foreign active GPU process in postflight")
    return {
        "schema_version": 1,
        "kind": "unscored_gdr_one_call_dispatch_trace",
        "mode": args.mode,
        "case_id": case["id"],
        "batch": case["batch"],
        "expected_grid_blocks": case["batch"] * 32 * 4,
        "expected_workgroup_threads": 4 * 64,
        "correctness_and_guards": "pass",
        "aiter_sha": AITERSHA,
        "spec_raw_sha256": SPEC_SHA256,
        "fixture_raw_sha256": raw_sha256(args.fixture),
        "source_raw_sha256": SOURCE_SHA256,
        "candidate_binary_raw_sha256": BINARY_SHA256,
        "scored_agent_candidate": False,
        "scored_eligible": False,
        "withheld_cases_evaluated": False,
        "host_pid_attributed": True,
        "foreign_active_gpu_processes": 0,
        "environment": environment,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("aiter", "hip"))
    parser.add_argument("--case", required=True, choices=BUCKETS)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    for field in ("spec", "fixture", "source", "candidate", "aiter_source", "host_gpu_report", "receipt"):
        setattr(args, field, getattr(args, field).resolve())
    code_roots = (args.spec.parent.parent, args.fixture.parents[2])
    if any(args.receipt.is_relative_to(root) for root in code_roots) or args.receipt.exists():
        parser.error("private receipt must be a new path outside repository")
    result = run(args)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    args.receipt.chmod(0o444)
    print(json.dumps({"case": args.case, "mode": args.mode, "correctness": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
