"""Unscored one-case graph feasibility on the nine admitted OPUS public shapes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tasks.opus_a16w16_persistent import admit as base_admit
from tasks.opus_a16w16_persistent.admit import sha256_file

sys.path.insert(0, str(Path(base_admit.__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adversarial_admit as admission
import feasibility


def classify(entry: dict, result: dict) -> dict:
    graph = entry.get("graph_performance", {})
    correct = bool(result.get("all_correct"))
    noise = bool(result.get("all_noise_qualified"))
    ratio = graph.get("adapter_to_aiter_median_ratio")
    return {
        "correctness_pass": correct,
        "noise_qualified": noise,
        "noninferiority_1p05": bool(correct and noise and isinstance(ratio, (int, float)) and ratio <= 1.05),
        "performance_interpretation": "graph_feasibility_only" if correct and noise else "inconclusive",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("matrix", "anchors", "aiter-source", "adapter-source", "adapter", "host-gpu-report", "case", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path if field != "case" else str)
    args = parser.parse_args()
    for field in ("matrix", "anchors", "aiter_source", "adapter_source", "adapter", "host_gpu_report", "output"):
        setattr(args, field, getattr(args, field).resolve())
    source_root = Path(__file__).resolve().parents[2]
    if args.output.exists() or args.output.is_relative_to(source_root) or args.output.is_relative_to(args.aiter_source):
        parser.error("output must be a new private file outside source trees")
    matrix = admission.check_provenance(args)
    case = admission.select_public_case(matrix, args.case)
    if admission._gpu_pids():
        raise RuntimeError("foreign active GPU process at preflight")
    sys.path.insert(0, str(args.aiter_source))
    import torch

    environment = admission._gpu_attestation(torch, args.host_gpu_report.read_text(encoding="utf-8"))
    result = feasibility.run({**matrix, "cases": [case]}, args.adapter)
    if len(result["cases"]) != 1 or result["cases"][0]["id"] != case["id"]:
        raise RuntimeError("feasibility route did not return the requested case")
    active = admission._gpu_pids()
    host_pid_attributed = os.getpid() in active and not any(pid != os.getpid() for pid in active)
    entry = result["cases"][0]
    result.update({
        "schema": "aiter-rs-opus-adversarial-public-graph-v1",
        "kind": "unscored_analyst_graph_feasibility",
        "case_id": case["id"],
        "input_pattern": "torch_random_seeded_for_shape_performance_control",
        "environment": environment,
        "aiter_sha": admission.AITERSHA,
        "matrix_sha256": admission.MATRIX_SHA,
        "adapter_source_sha256": admission.ADAPTER_SOURCE_SHA,
        "adapter_binary_sha256": admission.ADAPTER_BINARY_SHA,
        "driver_sha256": sha256_file(Path(__file__)),
        "host_gpu_report_sha256": sha256_file(args.host_gpu_report),
        "host_pid_attributed": host_pid_attributed,
        "withheld_cases_evaluated": False,
        "scored_agent_candidate": False,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    result["gate"] = classify(entry, result)
    result["gate"]["host_pid_attributed"] = host_pid_attributed
    if not host_pid_attributed:
        result["gate"]["performance_interpretation"] = "inconclusive"
        result["gate"]["noninferiority_1p05"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({
        "case": args.case,
        "correct": result["gate"]["correctness_pass"],
        "noise_qualified": result["gate"]["noise_qualified"],
        "ratio": entry.get("graph_performance", {}).get("adapter_to_aiter_median_ratio"),
    }))
    return 0 if result["gate"]["correctness_pass"] and host_pid_attributed else 1


if __name__ == "__main__":
    raise SystemExit(main())
