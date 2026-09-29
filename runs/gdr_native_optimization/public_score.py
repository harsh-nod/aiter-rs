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


def run_public(args) -> dict:
    task, contract = validate_task(args.task_dir)
    repo = args.task_dir.parents[2]
    git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
    subprocess.run(
        [*git, "diff", "--quiet", task["harness_revision"], "HEAD", "--", "harness", "references"],
        check=True, timeout=30,
    )
    subprocess.run(
        [*git, "diff", "--quiet", "--", "harness", "references"],
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

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != task["target_arch"]:
        raise RuntimeError(f"GPU arch mismatch: {arch}")
    sys.path.insert(0, str(args.aiter_source))
    spec = read_spec(args.snapshot / "public/spec.json")
    fixture = json.loads((args.snapshot / "public/large_fixture.json").read_text())["case"]
    if [case["id"] for case in spec["cases"]] + [fixture["id"]] != contract["visible_correctness_case_ids"]:
        raise ValueError("public cases differ from contract")
    plugin = importlib.import_module(spec["plugin"])
    candidate = plugin.load_hip_candidate(library)
    cases = spec["cases"] + [fixture]
    checked = [run_case(plugin, candidate, case) for case in cases]
    raw["arch"] = arch
    raw["candidate_binary_sha256"] = sha256(library)
    raw["correctness_details"] = checked
    raw["visible_case_results"] = {entry["id"]: bool(entry["pass"]) for entry in checked}
    if not all(entry["pass"] for entry in checked):
        raw.update({"status": "correctness_failed", "benchmark_bucket_results": {}})
        return raw
    if args.kind == "correctness":
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
    parser.add_argument("--kind", choices=("correctness", "benchmark"), required=True)
    args = parser.parse_args()
    for name in ("snapshot", "aiter_source", "task_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    repo = Path(__file__).resolve().parents[2]
    if args.output.is_relative_to(repo) or args.output.is_relative_to(args.snapshot):
        parser.error("raw public result must be outside code and snapshot")
    if not args.output.is_dir() or any(args.output.iterdir()):
        parser.error("raw public output must be an existing empty host-mounted directory")
    try:
        raw = run_public(args)
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
