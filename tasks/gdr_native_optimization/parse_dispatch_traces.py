"""Validate six private rocprofv3 GDR traces into a sanitized dispatch receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


JOBS = (
    ("valid_aiter_04", "aiter", "valid_slots"),
    ("valid_hip_01", "hip", "valid_slots"),
    ("strided_aiter_01", "aiter", "strided_mixed"),
    ("strided_hip_01", "hip", "strided_mixed"),
    ("large_aiter_01", "aiter", "large_valid_slots_candidate"),
    ("large_hip_01", "hip", "large_valid_slots_candidate"),
)
KERNEL_SUBSTRING = "gdr_decode_packed_bf16_kernel"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(paths: list[Path]) -> list[tuple[Path, dict]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend((path, row) for row in csv.DictReader(handle))
    return rows


def parse_one(root: Path, tag: str, mode: str, case_id: str) -> dict:
    receipt_path = root / f"receipt_{tag}.json"
    driver = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (driver["mode"], driver["case_id"], driver["correctness_and_guards"]) != (mode, case_id, "pass"):
        raise ValueError(f"driver correctness/provenance mismatch for {tag}")
    profile = root / f"profile_{tag}"
    kernel_files = sorted(profile.rglob("*_kernel_trace.csv"))
    api_files = sorted(profile.rglob("*_hip_api_trace.csv"))
    if not kernel_files or not api_files:
        raise ValueError(f"missing profiler CSV for {tag}")
    targets = [
        (path, row) for path, row in csv_rows(kernel_files)
        if KERNEL_SUBSTRING in row.get("Kernel_Name", "")
    ]
    if len(targets) != 1:
        raise ValueError(f"expected exactly one GDR kernel dispatch for {tag}; got {len(targets)}")
    kernel_path, kernel = targets[0]
    expected_threads = driver["expected_grid_blocks"] * driver["expected_workgroup_threads"]
    dims = tuple(int(kernel[name]) for name in (
        "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
        "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
    ))
    if dims != (256, 1, 1, expected_threads, 1, 1):
        raise ValueError(f"GDR launch geometry mismatch for {tag}: {dims}")
    correlations = [
        (path, row) for path, row in csv_rows(api_files)
        if row.get("Correlation_Id") == kernel["Correlation_Id"]
        and row.get("Process_Id") == kernel["Thread_Id"]
    ]
    if len(correlations) != 1 or correlations[0][1].get("Function") != "hipLaunchKernel":
        raise ValueError(f"GDR dispatch has no unique hipLaunchKernel correlation for {tag}")
    api_path, api = correlations[0]
    return {
        "tag": tag,
        "mode": mode,
        "case_id": case_id,
        "batch": driver["batch"],
        "kernel_name": kernel["Kernel_Name"],
        "launch_api": api["Function"],
        "target_dispatch_count": 1,
        "grid_blocks": driver["expected_grid_blocks"],
        "grid_work_items": expected_threads,
        "workgroup_threads": 256,
        "driver_receipt_sha256": file_sha256(receipt_path),
        "kernel_csv_sha256": file_sha256(kernel_path),
        "api_csv_sha256": file_sha256(api_path),
        "correctness_and_guards": "pass",
    }


def parse_all(root: Path) -> dict:
    entries = [parse_one(root, *job) for job in JOBS]
    if len({entry["kernel_name"] for entry in entries}) != 1:
        raise ValueError("AITER/HIP kernel identity differs across traces")
    return {
        "schema_version": 1,
        "kind": "unscored_gdr_profiler_dispatch_receipt",
        "scored_agent_candidate": False,
        "withheld_cases_evaluated": False,
        "source": "rocprofv3_1.1.0_kernel_and_hip_runtime_csv",
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.root.resolve()):
        parser.error("output must be a new private path under the trace root")
    receipt = parse_all(args.root.resolve())
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    output.chmod(0o444)
    print(json.dumps({"entries": len(receipt["entries"]), "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
