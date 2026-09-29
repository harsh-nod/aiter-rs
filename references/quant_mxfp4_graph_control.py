"""Unscored graph-replay control for the pinned MXFP4 Even r001 HIP binary."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from harness.core import sha256_path
from harness.run import _active_gpu_pids, _command, _gpu_manifest, _timed_call


R001_RAW_SHA256 = "757d0b4fd475a8c58ed4e2e1884ff233c1654be28245e82f79d5b0d0f2b0f7c3"
R001_PATH_SHA256 = "bcb1fa28074017435e3cb535623cae4d7d161f5a2591c4d5ff1a1890bf457d1b"
SPEC_RAW_SHA256 = "127e3bc54b2fbd312b5b34d3d95526bbc7603e148a854aa3ed5de9ec986a6d61"
GRAPH_CALLS = 32
WARMUP = 5
REPEATS = 20
GUARD = 256


class GuardedInput:
    def __init__(self, value, device=None):
        import torch

        target = device or value.device
        nbytes = value.numel() * value.element_size()
        self.raw = torch.full((nbytes + 768,), 0xA5, dtype=torch.uint8, device=target)
        self.start = GUARD + (-(self.raw.data_ptr() + GUARD) % GUARD)
        self.end = self.start + nbytes
        self.tensor = self.raw[self.start:self.end].view(value.dtype).reshape(value.shape)
        self.tensor.copy_(value)
        self.initial = self.raw[self.start:self.end].clone()

    def check(self):
        import torch

        return {
            "prefix_guard": bool(torch.all(self.raw[:self.start] == 0xA5)),
            "suffix_guard": bool(torch.all(self.raw[self.end:] == 0xA5)),
            "input_unchanged": bool(torch.equal(self.raw[self.start:self.end], self.initial)),
        }


def exact_bytes(observed, expected):
    if len(observed) != len(expected):
        return {"pass": False, "expected_bytes": len(expected), "observed_bytes": len(observed)}
    offsets = [index for index, (left, right) in enumerate(zip(observed, expected)) if left != right]
    return {"pass": not offsets, "mismatch_count": len(offsets), "first_mismatch_offsets": offsets[:8]}


def capture_graph(torch, fn, calls=GRAPH_CALLS):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    return graph


def paired_samples(timed, baseline, candidate, repeats=REPEATS):
    samples = {"aiter_ms": [], "candidate_ms": []}
    for index in range(repeats):
        order = (("aiter_ms", baseline), ("candidate_ms", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for key, fn in order:
            samples[key].append(timed(fn) / GRAPH_CALLS)
    return samples


def summarize_samples(samples, max_mad_ratio=0.05, threshold_ratio=1.05):
    baseline = samples["aiter_ms"]
    proposed = samples["candidate_ms"]
    if len(baseline) != len(proposed) or len(baseline) < 3 or min(baseline + proposed) <= 0:
        raise ValueError("positive paired samples required")
    base_median = statistics.median(baseline)
    candidate_median = statistics.median(proposed)
    mad = statistics.median(abs(value - base_median) for value in baseline) / base_median
    p95_index = int(0.95 * len(baseline) + 0.999999) - 1
    ratio = candidate_median / base_median
    return {
        **samples,
        "method": "32_call_graph_replay",
        "aiter_median_ms": base_median,
        "candidate_median_ms": candidate_median,
        "aiter_p95_ms": sorted(baseline)[p95_index],
        "candidate_p95_ms": sorted(proposed)[p95_index],
        "baseline_relative_mad": mad,
        "median_ratio": ratio,
        "noise_qualified": mad <= max_mad_ratio,
        "exploratory_threshold_pass": mad <= max_mad_ratio and ratio <= threshold_ratio,
    }


def quiet_gpu():
    report = _command("rocm-smi", "--showpids")
    if "KFD process information" not in report:
        raise RuntimeError("GPU PID probe unavailable")
    pids = _active_gpu_pids(report)
    if os.getpid() not in pids:
        raise RuntimeError("scorer host PID not visible in GPU process table; use --pid=host")
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign:
        raise RuntimeError(f"foreign active GPU processes: {foreign}")
    return {"self_pid_visible": True, "foreign_active_count": 0}


def reset_output(output):
    import torch

    output.packed.view(torch.uint8).fill_(0xA5)
    output.scales.view(torch.uint8).fill_(0xA5)


def compare_pair(plugin, output, oracle_pair, baseline_pair=None):
    packed, scales = plugin.output_bytes(output)
    comparison = {
        "oracle_packed": exact_bytes(packed, oracle_pair[0]),
        "oracle_scale": exact_bytes(scales, oracle_pair[1]),
    }
    if baseline_pair is not None:
        comparison["aiter_packed"] = exact_bytes(packed, baseline_pair[0])
        comparison["aiter_scale"] = exact_bytes(scales, baseline_pair[1])
    comparison["pass"] = all(value["pass"] for value in comparison.values())
    return comparison


def check_case(plugin, candidate, case):
    import torch

    source = plugin.make_input(case)
    guarded = GuardedInput(source)
    x = guarded.tensor
    oracle_pair = plugin.oracle(x)
    aiter_out = plugin.allocate_outputs(x)
    candidate_out = plugin.allocate_outputs(x)

    def aiter_call():
        from aiter.ops.quant import quant_mxfp4

        quant_mxfp4(x, aiter_out.packed, aiter_out.scales, 32, 2, False, False, False, False)

    def candidate_call():
        candidate.run_into(x, candidate_out)

    repetitions = []
    for _ in range(2):
        reset_output(aiter_out)
        reset_output(candidate_out)
        aiter_call()
        candidate_call()
        torch.cuda.synchronize()
        plugin.validate_output(aiter_out, x)
        plugin.validate_output(candidate_out, x, aiter_out)
        baseline_pair = plugin.output_bytes(aiter_out)
        aiter_result = compare_pair(plugin, aiter_out, oracle_pair)
        candidate_result = compare_pair(plugin, candidate_out, oracle_pair, baseline_pair)
        input_check = guarded.check()
        repetitions.append({
            "aiter": aiter_result, "candidate": candidate_result,
            "input": input_check,
            "pass": aiter_result["pass"] and candidate_result["pass"] and all(input_check.values()),
        })
    return {
        "id": case["id"], "bucket": case["bucket"],
        "correctness": {"pass": all(item["pass"] for item in repetitions), "repetitions": repetitions},
    }, (guarded, oracle_pair, aiter_out, candidate_out, aiter_call, candidate_call)


def benchmark_case(torch, plugin, state, spec):
    guarded, oracle_pair, aiter_out, candidate_out, aiter_call, candidate_call = state
    preflight = quiet_gpu()
    for _ in range(WARMUP):
        aiter_call()
        candidate_call()
    torch.cuda.synchronize()
    aiter_graph = capture_graph(torch, aiter_call)
    candidate_graph = capture_graph(torch, candidate_call)
    torch.cuda.synchronize()
    samples = paired_samples(_timed_call, aiter_graph.replay, candidate_graph.replay)
    torch.cuda.synchronize()
    postflight = quiet_gpu()
    plugin.validate_output(aiter_out, guarded.tensor)
    plugin.validate_output(candidate_out, guarded.tensor, aiter_out)
    baseline_pair = plugin.output_bytes(aiter_out)
    checks = {
        "aiter": compare_pair(plugin, aiter_out, oracle_pair),
        "candidate": compare_pair(plugin, candidate_out, oracle_pair, baseline_pair),
        "input": guarded.check(),
    }
    checks["pass"] = checks["aiter"]["pass"] and checks["candidate"]["pass"] and all(checks["input"].values())
    summary = summarize_samples(samples, spec["max_mad_ratio"], spec["threshold_ratio"])
    summary["preflight"] = preflight
    summary["postflight"] = postflight
    summary["post_graph_checks"] = checks
    if not checks["pass"]:
        summary["exploratory_threshold_pass"] = False
    return summary


def run(args):
    import torch

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if hashlib.sha256(args.spec.read_bytes()).hexdigest() != SPEC_RAW_SHA256:
        raise RuntimeError("public quant spec changed")
    revision = subprocess.check_output(["git", "-C", str(args.aiter_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["aiter_sha"]:
        raise RuntimeError("pinned AITER revision mismatch")
    raw_sha = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    path_sha = sha256_path(args.candidate)
    if (raw_sha, path_sha) != (R001_RAW_SHA256, R001_PATH_SHA256):
        raise RuntimeError("candidate is not the admitted r001 binary")
    sys.path.insert(0, str(args.aiter_source))
    importlib.import_module("aiter")
    plugin = importlib.import_module(spec["plugin"])
    environment = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    candidate = plugin.load_hip_candidate(args.candidate)
    public = [case for case in spec["cases"] if case.get("visibility") == "visible"]
    if len(public) != len(spec["cases"]):
        raise RuntimeError("graph control accepts public cases only")
    entries = []
    for case in public:
        try:
            entry, state = check_case(plugin, candidate, case)
            if entry["correctness"]["pass"] and case["id"] in spec["benchmark_case_ids"]:
                entry["graph_timing"] = benchmark_case(torch, plugin, state, spec)
            entries.append(entry)
        except Exception as exc:
            entries.append({"id": case["id"], "error": repr(exc), "traceback": traceback.format_exc(), "correctness": {"pass": False}})
    full_correct = len(entries) == len(public) and all(entry["correctness"]["pass"] for entry in entries)
    timed = [entry for entry in entries if "graph_timing" in entry]
    return {
        "schema_version": 1, "task_id": spec["task_id"], "control_kind": "r001_unscored_graph_replay",
        "scored": False, "scored_eligible": False, "withheld_cases_evaluated": False,
        "aiter_sha": revision, "spec_raw_sha256": SPEC_RAW_SHA256,
        "candidate_raw_sha256": raw_sha, "candidate_filename_inclusive_sha256": path_sha,
        "environment": environment,
        "operator_boundary": "preallocated low_level_quant_mxfp4_even_no_shuffle",
        "host_gap_caveat": "Graph replay amortizes Python enqueue gaps; legacy Python-call GPU events are not device-kernel latency evidence.",
        "graph_calls": GRAPH_CALLS, "paired_repeats": REPEATS, "warmup": WARMUP,
        "correctness": {"status": "complete_pass" if full_correct else "fail", "case_count": len(entries), "cases": entries},
        "performance": {"status": "exploratory_graph" if len(timed) == len(spec["benchmark_case_ids"]) else "not_run_or_partial",
                        "benchmark_case_count": len(timed),
                        "exploratory_joint_threshold_pass": full_correct and len(timed) == len(spec["benchmark_case_ids"]) and all(
                            entry["graph_timing"]["exploratory_threshold_pass"] for entry in timed)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unscored-r001-control", action="store_true", required=True)
    args = parser.parse_args()
    if not args.unscored_r001_control:
        parser.error("this control is unscored")
    for name in ("spec", "candidate", "aiter_source", "host_gpu_report", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.is_relative_to(Path.cwd().resolve()):
        parser.error("raw result must be outside repository")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args)
    except Exception as exc:
        result = {"schema_version": 1, "control_kind": "r001_unscored_graph_replay",
                  "scored": False, "scored_eligible": False,
                  "error": repr(exc), "traceback": traceback.format_exc(),
                  "correctness": {"status": "setup_error"}, "performance": {"status": "not_run"}}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = args.output / "result.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)
    print(json.dumps({"correctness": result["correctness"]["status"],
                      "performance": result["performance"]["status"], "result": str(path)}))
    return 0 if result["correctness"]["status"] == "complete_pass" and result["performance"]["status"] == "exploratory_graph" else 1


if __name__ == "__main__":
    raise SystemExit(main())
