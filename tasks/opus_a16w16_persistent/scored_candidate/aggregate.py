"""Trusted exact-matrix OPUS production-overlay gate; raw inputs stay private."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .freeze import HERE, PUBLIC, REPO, performance_gate, sha256, validate_matrix, validate_task


def load_results(root: Path) -> list[dict]:
    paths = sorted(root.glob("*/result.json"))
    if not paths:
        raise ValueError("no per-case production overlay results")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _validate_provenance(result: dict, task: dict, matrix_sha: str, candidate_sha: str) -> None:
    if (result.get("schema") != "aiter-rs-opus-production-overlay-pair-v1"
            or result.get("kind") != "unscored_production_overlay_control"
            or result.get("scored_eligible") is not False
            or result.get("aiter_sha") != task["aiter_sha"]
            or result.get("task_sha256") != sha256(HERE / "task.json")
            or result.get("matrix_sha256") != matrix_sha
            or result.get("driver_sha256") != sha256(HERE / "dual_worker.py")
            or result.get("host_gpu_report_sha256") != task["host_gpu_report_sha256"]
            or result.get("compiler_image_id") != task["compiler_image_id"]):
        raise ValueError("per-case scorer provenance does not match frozen task")
    source = result.get("source_check", {})
    if (source.get("baseline_header_sha256") != task["base_header_sha256"]
            or source.get("candidate_header_sha256") != candidate_sha
            or source.get("overlay_changed_paths") not in ([], [task["editable_path"]])):
        raise ValueError("per-case candidate header/source changed")
    for role in ("aiter", "candidate"):
        entry = result.get(role, {})
        if (entry.get("correctness_pass") is not True
                or entry.get("exact_dispatch") is not True
                or entry.get("resolved_kid") not in (300, 1300)
                or entry.get("workspace_used") is not False
                or not isinstance(entry.get("jit_module_sha256"), str)):
            raise ValueError("per-case production dispatch or correctness failed")


def aggregate_public(results: list[dict], task: dict, candidate_sha: str) -> dict:
    ids = [result.get("case_id") for result in results]
    if len(results) != len(PUBLIC) or set(ids) != set(PUBLIC) or len(set(ids)) != len(ids):
        raise ValueError("exact eight public case results required")
    rows = []
    for result in results:
        _validate_provenance(result, task, task["public_matrix_sha256"], candidate_sha)
        if (result.get("status") not in {"single_bucket_pass", "single_bucket_performance_failed"}
                or result.get("same_boundary") is not True
                or any(result.get(role, {}).get("pre_graph", {}).get("pass") is not True for role in ("aiter", "candidate"))
                or any(result.get(role, {}).get("pre_timing", {}).get("pass") is not True for role in ("aiter", "candidate"))
                or result["aiter"].get("performance_input_hashes") != result["candidate"].get("performance_input_hashes")
                or result.get("post_graph", {}).get("aiter", {}).get("pass") is not True
                or result.get("post_graph", {}).get("candidate", {}).get("pass") is not True):
            raise ValueError("public case lacks guarded production graph evidence")
        expected_kid = PUBLIC[result["case_id"]][3]
        if any(result[role]["resolved_kid"] != expected_kid for role in ("aiter", "candidate")):
            raise ValueError("per-case kid changed")
        graph = result.get("graph_performance", {})
        rows.append({
            "id": result["case_id"], "candidate_to_aiter_ratio": graph.get("candidate_to_aiter_ratio"),
            "aiter_relative_mad": graph.get("aiter_relative_mad"),
            "candidate_relative_mad": graph.get("candidate_relative_mad"),
            "correctness_pass": True, "post_graph_readonly_pass": True,
            "exact_dispatch": True, "same_boundary": True,
            "pid_gate": result.get("pid_gate"),
        })
    return performance_gate(rows)


def aggregate_withheld(results: list[dict], matrix: dict, task: dict, candidate_sha: str) -> dict:
    validate_matrix(matrix, withheld=True)
    expected = {case["id"]: case["kid"] for case in matrix["cases"]}
    ids = [result.get("case_id") for result in results]
    if len(results) != len(expected) or set(ids) != set(expected) or len(set(ids)) != len(ids):
        raise ValueError("exact committed withheld case results required")
    for result in results:
        _validate_provenance(result, task, task["withheld_matrix_sha256"], candidate_sha)
        if result.get("status") != "single_withheld_case_correctness_pass" or result.get("pass") is not True or result.get("pid_gate") is not True:
            raise ValueError("withheld correctness or PID gate failed")
        if any(result[role]["resolved_kid"] != expected[result["case_id"]] for role in ("aiter", "candidate")):
            raise ValueError("withheld exact kid changed")
    return {"case_count": len(expected), "correctness_pass": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--withheld-root", type=Path)
    parser.add_argument("--withheld-matrix", type=Path)
    parser.add_argument("--candidate-header-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.withheld_root) != bool(args.withheld_matrix):
        parser.error("withheld root and matrix must be supplied together")
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(REPO) or output.parent.stat().st_mode & 0o077:
        parser.error("aggregate raw output must be new under a mode-700 private directory")
    task = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    public_matrix = json.loads((HERE / "public_matrix.json").read_text(encoding="utf-8"))
    validate_task(task, public_matrix)
    public = aggregate_public(load_results(args.public_root), task, args.candidate_header_sha256)
    hidden = None
    if args.withheld_root:
        path = args.withheld_matrix.resolve()
        if path.is_relative_to(REPO) or sha256(path) != task["withheld_matrix_sha256"]:
            raise ValueError("private matrix location or commitment mismatch")
        hidden = aggregate_withheld(load_results(args.withheld_root), json.loads(path.read_text()), task, args.candidate_header_sha256)
    report = {"schema": "aiter-rs-opus-production-overlay-aggregate-v1",
              "task_sha256": sha256(HERE / "task.json"), "candidate_header_sha256": args.candidate_header_sha256,
              "public": public, "withheld": hidden, "scored_eligible": False,
              "ready_for_review": public["per_bucket_noninferior"] and bool(hidden and hidden["correctness_pass"])}
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"public_case_count": len(PUBLIC), "public_noninferior": public["per_bucket_noninferior"],
                      "geomean_ratio": public["geomean_ratio"],
                      "withheld_correctness_pass": bool(hidden and hidden["correctness_pass"]),
                      "ready_for_review": report["ready_for_review"], "raw_sha256": sha256(output)}))
    return 0 if report["ready_for_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
