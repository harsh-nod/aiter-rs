"""Unscored gfx950 graph-replay feasibility probe for sparse prefill.

This compares the pinned AITER low-level dispatch with a HIP analyst control.
It is not an agent score. Run with an external watchdog; native HIP is trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from references.sparse_prefill_admission import GUARD
from references.sparse_prefill_gfx950 import AITER_SHA, Case, make_inputs, oracle, public_cases
from references.sparse_prefill_scorer import HipCandidate, POISON, compare_output


BUCKET_CASE_IDS = ("mixed_sources", "tile_tail_63_64_65")
CAPTURED_CALLS = 32
WARMUP_REPLAYS = 5
ALTERNATING_PAIRS = 20
MAX_LATENCY_RATIO = 1.05
MAX_MAD_RATIO = 0.05


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases() -> tuple[Case, ...]:
    cases = {case.name: case for case in public_cases()}
    if len(cases) != len(public_cases()):
        raise ValueError("public case names are not unique")
    return tuple(cases[name] for name in BUCKET_CASE_IDS)


def summarize_samples(aiter_ms: list[float], candidate_ms: list[float]) -> dict:
    if len(aiter_ms) != ALTERNATING_PAIRS or len(candidate_ms) != ALTERNATING_PAIRS:
        raise ValueError("incomplete alternating timing pairs")
    if any(not (0 < sample < float("inf")) for sample in aiter_ms + candidate_ms):
        raise ValueError("nonpositive or nonfinite timing sample")
    aiter_median = statistics.median(aiter_ms)
    candidate_median = statistics.median(candidate_ms)
    aiter_mad_ratio = statistics.median(abs(x - aiter_median) for x in aiter_ms) / aiter_median
    candidate_mad_ratio = (
        statistics.median(abs(x - candidate_median) for x in candidate_ms) / candidate_median
    )
    ratio = candidate_median / aiter_median
    noise_qualified = max(aiter_mad_ratio, candidate_mad_ratio) <= MAX_MAD_RATIO
    return {
        "aiter_median_ms_per_call": aiter_median,
        "candidate_median_ms_per_call": candidate_median,
        "candidate_over_aiter": ratio,
        "aiter_mad_ratio": aiter_mad_ratio,
        "candidate_mad_ratio": candidate_mad_ratio,
        "noise_qualified": noise_qualified,
        "within_1p05": ratio <= MAX_LATENCY_RATIO,
        "pass": noise_qualified and ratio <= MAX_LATENCY_RATIO,
    }


def inspect_output(gpu_inputs: dict, cpu_inputs: dict, storage: torch.Tensor,
                   expected: torch.Tensor, return_code: int = 0) -> dict:
    output = storage[GUARD:-GUARD].view_as(expected)
    return compare_output(
        output.cpu(), expected,
        guard_before=storage[:GUARD].cpu(),
        guard_after=storage[-GUARD:].cpu(),
        inputs_unchanged=all(
            torch.equal(tensor.cpu(), cpu_inputs[name]) for name, tensor in gpu_inputs.items()
        ),
        return_code=return_code,
    )


def checked_output(call, gpu_inputs: dict, cpu_inputs: dict, storage: torch.Tensor,
                   expected: torch.Tensor) -> dict:
    storage[GUARD:-GUARD].view_as(expected).fill_(POISON)
    code = call()
    torch.cuda.synchronize()
    return inspect_output(gpu_inputs, cpu_inputs, storage, expected, code)


def elapsed_ms(graph: torch.cuda.CUDAGraph, start: torch.cuda.Event,
               end: torch.cuda.Event) -> float:
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / CAPTURED_CALLS


def benchmark_case(case: Case, op, candidate: HipCandidate) -> dict:
    cpu_inputs = make_inputs(case)
    expected = oracle(cpu_inputs, case.softmax_scale)
    gpu_inputs = {name: tensor.cuda() for name, tensor in cpu_inputs.items()}
    storages = {
        name: torch.full((expected.numel() + 2 * GUARD,), POISON,
                         dtype=torch.bfloat16, device="cuda")
        for name in ("aiter", "candidate")
    }
    outputs = {name: storage[GUARD:-GUARD].view_as(expected)
               for name, storage in storages.items()}

    def run_aiter() -> int:
        op.pa_sparse_prefill_gfx950_opus_fwd(
            gpu_inputs["q"], gpu_inputs["unified_kv"],
            gpu_inputs["kv_indices_prefix"], gpu_inputs["kv_indptr_prefix"],
            gpu_inputs["kv"], gpu_inputs["kv_indices_extend"],
            gpu_inputs["kv_indptr_extend"], gpu_inputs["attn_sink"],
            outputs["aiter"], float(case.softmax_scale),
        )
        return 0

    def run_candidate() -> int:
        return candidate.run(gpu_inputs, outputs["candidate"], case.softmax_scale)

    calls = {"aiter": run_aiter, "candidate": run_candidate}
    before = {
        name: checked_output(call, gpu_inputs, cpu_inputs, storages[name], expected)
        for name, call in calls.items()
    }
    if not all(result["passed"] for result in before.values()):
        return {"status": "correctness_failed_before", "before": before}

    graphs = {}
    for name, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.graph(graph):
            for _ in range(CAPTURED_CALLS):
                if call() != 0:
                    raise RuntimeError(f"{name} launch failed during capture")
        graphs[name] = (graph, start, end)

    for _ in range(WARMUP_REPLAYS):
        for graph, _, _ in graphs.values():
            graph.replay()
        torch.cuda.synchronize()

    samples = {"aiter_ms_per_call": [], "candidate_ms_per_call": []}
    for pair in range(ALTERNATING_PAIRS):
        order = ("aiter", "candidate") if pair % 2 == 0 else ("candidate", "aiter")
        for name in order:
            samples[f"{name}_ms_per_call"].append(elapsed_ms(*graphs[name]))

    after = {
        name: inspect_output(gpu_inputs, cpu_inputs, storages[name], expected)
        for name in calls
    }
    if not all(result["passed"] for result in after.values()):
        return {
            "status": "correctness_failed_after", "before": before, "after": after,
            "after_check": "read_only_post_graph_replay",
        }
    return {
        "status": "complete", "before": before, "after": after,
        "after_check": "read_only_post_graph_replay",
        "samples": samples,
        "summary": summarize_samples(
            samples["aiter_ms_per_call"], samples["candidate_ms_per_call"]
        ),
    }


def run_probe(args) -> dict:
    if sha256_file(args.candidate) != args.candidate_sha256:
        raise ValueError("candidate binary differs from preregistered SHA256")
    if sha256_file(args.candidate_source) != args.candidate_source_sha256:
        raise ValueError("candidate source differs from preregistered SHA256")
    revision = subprocess.check_output(
        ["git", "-C", str(args.aiter_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AITER_SHA:
        raise ValueError(f"AITER revision mismatch: {revision}")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU unavailable")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"expected gfx950, found {arch}")
    sys.path.insert(0, str(args.aiter_source))
    op = importlib.import_module("aiter.ops.pa_sparse_prefill_opus")
    if Path(op.__file__).resolve() != (args.aiter_source / "aiter/ops/pa_sparse_prefill_opus.py").resolve():
        raise RuntimeError("AITER module is not from pinned checkout")
    if op.get_gfx_runtime() != "gfx950":
        raise RuntimeError("AITER dispatch does not report gfx950")
    candidate = HipCandidate(args.candidate)
    buckets = {case.name: benchmark_case(case, op, candidate) for case in selected_cases()}
    complete = all(item["status"] == "complete" for item in buckets.values())
    return {
        "status": "complete" if complete else "correctness_failed",
        "arch": arch,
        "buckets": buckets,
        "joint_pass": complete and all(item["summary"]["pass"] for item in buckets.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--candidate-source", required=True, type=Path)
    parser.add_argument("--candidate-source-sha256", required=True)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for key in ("candidate", "candidate_source", "aiter_source", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    repo = Path(__file__).resolve().parents[2]
    if args.output_dir.is_relative_to(repo):
        parser.error("raw results must remain outside the repository")
    if args.output_dir.exists():
        parser.error("output directory already exists")

    result = {
        "schema_version": 1,
        "kind": "unscored_sparse_prefill_graph_feasibility",
        "aiter_sha": AITER_SHA,
        "candidate_binary_sha256": args.candidate_sha256,
        "candidate_source_sha256": args.candidate_source_sha256,
        "probe_source_sha256": sha256_file(Path(__file__)),
        "public_generator_sha256": sha256_file(Path(sys.modules[Case.__module__].__file__)),
        "buckets_selected": list(BUCKET_CASE_IDS),
        "captured_calls_per_graph": CAPTURED_CALLS,
        "warmup_replays": WARMUP_REPLAYS,
        "alternating_pairs": ALTERNATING_PAIRS,
        "max_latency_ratio": MAX_LATENCY_RATIO,
        "max_mad_ratio": MAX_MAD_RATIO,
        "timing_mode": "gpu_events_around_graph_replay",
        "joint_pass": False,
        "scored_agent_candidate": False,
    }
    try:
        result.update(run_probe(args))
    except Exception as exc:
        result.update({"status": "probe_error", "error": repr(exc)})
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, mode=0o700)
    output = args.output_dir / "result.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(result, file, sort_keys=True, indent=2)
        file.write("\n")
    print(json.dumps({
        "status": result["status"], "joint_pass": result["joint_pass"],
        "buckets": {
            name: item.get("summary", {}).get("pass")
            for name, item in result.get("buckets", {}).items()
        },
    }))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
