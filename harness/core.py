"""Pure scoring and artifact helpers used by the GPU harness."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median


def sha256_path(path: Path) -> str:
    """Hash a file or a directory tree with path names included."""
    path = path.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"no files to hash: {path}")
    digest = hashlib.sha256()
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "schema_version", "task_id", "aiter_sha", "gpu_sku", "gpu_pci_device_id",
        "target_arch", "plugin", "cases",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"missing spec keys: {', '.join(missing)}")
    if spec["schema_version"] != 1 or spec["target_arch"] != "gfx950":
        raise ValueError("only schema v1 gfx950 tasks are supported")
    if len(spec["aiter_sha"]) != 40 or any(c not in "0123456789abcdef" for c in spec["aiter_sha"]):
        raise ValueError("aiter_sha must be a lowercase 40-digit commit SHA")
    if not spec["cases"] or len({case["id"] for case in spec["cases"]}) != len(spec["cases"]):
        raise ValueError("cases must be nonempty and have unique IDs")
    if any(case.get("visibility") != "visible" for case in spec["cases"]):
        raise ValueError("public spec may contain only visible cases")
    commitment = spec.get("withheld_cases_sha256", "")
    if len(commitment) != 64 or any(c not in "0123456789abcdef" for c in commitment):
        raise ValueError("public spec must commit to withheld cases by SHA256")
    if not spec.get("threshold_ratio", 1.05) >= 1.0:
        raise ValueError("threshold_ratio must be at least 1.0")
    return spec


def merge_withheld(spec: dict, path: Path) -> dict:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != spec["withheld_cases_sha256"]:
        raise ValueError("withheld-case SHA256 does not match the public commitment")
    hidden = json.loads(raw)
    if hidden.get("schema_version") != 1 or hidden.get("task_id") != spec["task_id"]:
        raise ValueError("withheld-case manifest has wrong schema or task ID")
    cases = hidden.get("cases", [])
    if not cases or any(case.get("visibility") != "withheld" for case in cases):
        raise ValueError("private manifest needs withheld cases")
    ids = [case["id"] for case in spec["cases"] + cases]
    if len(ids) != len(set(ids)):
        raise ValueError("visible/withheld case IDs overlap")
    hidden_benchmarks = hidden.get("benchmark_case_ids", [])
    if any(case_id not in {case["id"] for case in cases} for case_id in hidden_benchmarks):
        raise ValueError("hidden benchmark IDs must name withheld cases")
    return {
        **spec,
        "cases": spec["cases"] + cases,
        "benchmark_case_ids": spec["benchmark_case_ids"] + hidden_benchmarks,
    }


def score_buckets(
    samples: dict[str, dict[str, list[float]]], threshold_ratio: float, max_mad_ratio: float = 0.05
) -> dict:
    """Require non-inferiority in every bucket, never only the aggregate."""
    scored = {}
    for bucket, variants in samples.items():
        baseline = variants["aiter_ms"]
        candidate = variants["candidate_ms"]
        if not baseline or not candidate:
            raise ValueError(f"missing timing samples in {bucket}")
        if any(not math.isfinite(x) or x <= 0 for x in baseline + candidate):
            raise ValueError(f"invalid timing sample in {bucket}")
        aiter_med = median(baseline)
        candidate_med = median(candidate)
        ratio = candidate_med / aiter_med
        aiter_mad = median(abs(value - aiter_med) for value in baseline) / aiter_med
        candidate_mad = median(abs(value - candidate_med) for value in candidate) / candidate_med
        qualified = max(aiter_mad, candidate_mad) <= max_mad_ratio
        scored[bucket] = {
            "aiter_median_ms": aiter_med,
            "candidate_median_ms": candidate_med,
            "ratio": ratio,
            "noise_qualified": qualified,
            "aiter_mad_ratio": aiter_mad,
            "candidate_mad_ratio": candidate_mad,
            "pass": qualified and ratio <= threshold_ratio,
            "aiter_samples_ms": baseline,
            "candidate_samples_ms": candidate,
        }
    return {"pass": bool(scored) and all(item["pass"] for item in scored.values()), "buckets": scored}
