"""Unscored gfx950 MHC candidate validation and same-boundary feasibility timing."""

from __future__ import annotations

import argparse
import ctypes
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

from harness.run import _active_gpu_pids, _command, _gpu_manifest, _timed_call
from mega.mhc_admit import OUTPUT_NAMES, compare_output
from mega.mhc_oracle import Parameters, make_inputs, post_pre, validate_case

INPUT_NAMES = (
    "layer_input", "residual_in", "post_layer_mix", "comb_res_mix",
    "fn", "hc_scale", "hc_base", "norm_weight",
)


class GuardedTensor:
    def __init__(self, shape, dtype, value=None, device="cuda"):
        import torch

        count = 1
        for dimension in shape:
            count *= dimension
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        self.raw = torch.empty(nbytes + 768, dtype=torch.uint8, device=device)
        self.raw.fill_(0xA5)
        self.start = 256 + (-(self.raw.data_ptr() + 256) % 256)
        self.end = self.start + nbytes
        self.tensor = self.raw[self.start:self.end].view(dtype).reshape(shape)
        if value is not None:
            self.tensor.copy_(value)
        self.original = self.raw[self.start:self.end].clone() if value is not None else None

    def guards_pass(self):
        import torch

        return bool(torch.all(self.raw[:self.start] == 0xA5)) and bool(
            torch.all(self.raw[self.end:] == 0xA5)
        )

    def input_unchanged(self):
        import torch

        return self.original is None or bool(torch.equal(self.raw[self.start:self.end], self.original))

    def reset_payload(self):
        self.raw[self.start:self.end].fill_(0xA5)


class Candidate:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.workspace_bytes = self.library.aiter_rs_mhc_workspace_bytes
        self.workspace_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        self.workspace_bytes.restype = ctypes.c_size_t
        self.launch = self.library.aiter_rs_mhc_post_pre
        self.launch.argtypes = (
            [ctypes.c_void_p] * 13
            + [ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64]
            + [ctypes.c_float] * 4
            + [ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        )
        self.launch.restype = ctypes.c_int

    def call(self, inputs, outputs, workspace, case, params):
        import torch

        pointers = [
            inputs[name].tensor.data_ptr() if name in inputs else None
            for name in INPUT_NAMES
        ]
        pointers += [outputs[name].tensor.data_ptr() for name in OUTPUT_NAMES]
        pointers += [workspace.tensor.data_ptr()]
        status = self.launch(
            *pointers, workspace.tensor.numel(), case["m"], case["hidden_size"],
            params.rms_eps, params.pre_eps, params.sinkhorn_eps,
            params.post_multiplier, params.sinkhorn_repeat, params.norm_eps,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"HIP candidate launch returned {status}")


def output_shapes(case):
    m, h = case["m"], case["hidden_size"]
    return {
        "post_mix": ((m, 4, 1), "float32"),
        "comb_mix": ((m, 4, 4), "float32"),
        "layer_input_out": ((m, h), "bfloat16"),
        "next_residual": ((m, 4, h), "bfloat16"),
    }


def make_outputs(case):
    import torch

    return {
        name: GuardedTensor(shape, getattr(torch, dtype))
        for name, (shape, dtype) in output_shapes(case).items()
    }


def make_gpu_inputs(cpu):
    return {
        name: GuardedTensor(tuple(value.shape), value.dtype, value)
        for name, value in cpu.items()
    }


def native_calls(module, case, inputs, outputs):
    import torch

    m, h = case["m"], case["hidden_size"]
    x = {name: item.tensor for name, item in inputs.items()}
    out = {name: item.tensor for name, item in outputs.items()}
    p = Parameters()
    if case["route"] == "fused":
        split, tile_m, tile_n, tile_k = module.get_mhc_fused_post_pre_config(m, h, False, False)
    else:
        split, tile_k = module.get_mhc_pre_splitk_large_m(m, 4 * h, w_preshuffle_bf16=False)
    pad = GuardedTensor((split, m, 32), torch.float32)
    sqr = GuardedTensor((split, m), torch.float32)
    gemm = pad.tensor[:, :, :24]

    def call():
        if case["route"] == "fused":
            module.mhc_fused_post_pre_gemm_sqrsum(
                gemm, sqr.tensor, out["next_residual"], x["layer_input"],
                x["residual_in"], x["post_layer_mix"], x["comb_res_mix"],
                x["fn"], tile_m, tile_n, tile_k,
                w_preshuffle_bf16=0, res_preshuffle=0,
            )
        else:
            module.mhc_post(
                out["next_residual"], x["layer_input"], x["residual_in"],
                x["post_layer_mix"], x["comb_res_mix"], 0,
            )
            module.mhc_pre_gemm_sqrsum(
                gemm, sqr.tensor, out["next_residual"], x["fn"], tile_k,
                w_preshuffle_bf16=0,
            )
        common = (
            out["post_mix"], out["comb_mix"], out["layer_input_out"],
            gemm, sqr.tensor, x["hc_scale"], x["hc_base"],
            out["next_residual"],
        )
        if case.get("norm"):
            module.mhc_pre_big_fuse_rmsnorm(
                *common, x["norm_weight"], p.rms_eps, p.pre_eps,
                p.sinkhorn_eps, p.norm_eps, p.post_multiplier,
                p.sinkhorn_repeat, res_preshuffle=0,
            )
        else:
            module.mhc_pre_big_fuse(
                *common, p.rms_eps, p.pre_eps, p.sinkhorn_eps,
                p.post_multiplier, p.sinkhorn_repeat, res_preshuffle=0,
            )

    labels = (
        ["mhc_fused_post_pre_gemm_sqrsum", "mhc_pre_big_fuse_rmsnorm" if case.get("norm") else "mhc_pre_big_fuse"]
        if case["route"] == "fused" else
        ["mhc_post", "mhc_pre_gemm_sqrsum", "mhc_pre_big_fuse"]
    )
    return call, (pad, sqr), labels


def compare_all(expected, outputs):
    return {
        name: compare_output(value, outputs[name].tensor.cpu(), name)
        for name, value in zip(OUTPUT_NAMES, expected, strict=True)
    }


def checks(inputs, outputs, workspace=None, scratch=()):
    input_checks = {name: {"guards": item.guards_pass(), "unchanged": item.input_unchanged()}
                    for name, item in inputs.items()}
    output_guards = {name: item.guards_pass() for name, item in outputs.items()}
    scratch_guards = [item.guards_pass() for item in scratch]
    workspace_guard = workspace.guards_pass() if workspace else None
    passed = (
        all(all(value.values()) for value in input_checks.values())
        and all(output_guards.values()) and all(scratch_guards)
        and (workspace_guard is None or workspace_guard)
    )
    return {
        "pass": passed, "inputs": input_checks, "outputs": output_guards,
        "scratch": scratch_guards, "workspace": workspace_guard,
    }


def run_case(module, candidate, case):
    import torch

    validate_case(case)
    p = Parameters()
    cpu = make_inputs(case)
    wanted = post_pre(cpu, p)
    inputs = make_gpu_inputs(cpu)
    native_out = make_outputs(case)
    native, scratch, labels = native_calls(module, case, inputs, native_out)
    native()
    torch.cuda.synchronize()
    native_comparison = compare_all(wanted, native_out)
    native_checks = checks(inputs, native_out, scratch=scratch)
    native_pass = all(entry["pass"] for entry in native_comparison.values()) and native_checks["pass"]
    entry = {
        "id": case["id"], "m": case["m"], "hidden_size": case["hidden_size"],
        "route": case["route"], "native_calls": labels,
        "aiter": {"outputs": native_comparison, "guards": native_checks, "pass": native_pass},
    }
    if not native_pass:
        entry["candidate"] = {"pass": None, "reason": "baseline_oracle_disagreement"}
        return entry, None

    need = candidate.workspace_bytes(case["m"], case["hidden_size"], bool(case.get("norm")))
    if need == 0 or need > 256 * 1024 * 1024:
        entry["candidate"] = {"pass": False, "reason": "invalid_workspace_size", "bytes": need}
        return entry, None
    workspace = GuardedTensor((need,), torch.uint8)
    candidate_out = make_outputs(case)

    def launch():
        candidate.call(inputs, candidate_out, workspace, case, p)

    repeats = []
    for _ in range(2):
        for item in candidate_out.values():
            item.reset_payload()
        workspace.reset_payload()
        try:
            launch()
            torch.cuda.synchronize()
        except Exception as exc:
            entry["candidate"] = {"pass": False, "reason": "launch_error", "error": repr(exc), "completed_repeats": repeats}
            return entry, None
        comparisons = compare_all(wanted, candidate_out)
        agreement = {
            name: compare_output(native_out[name].tensor.cpu(), candidate_out[name].tensor.cpu(), name)
            for name in OUTPUT_NAMES
        }
        guards = checks(inputs, candidate_out, workspace)
        passed = guards["pass"] and all(x["pass"] for x in comparisons.values()) and all(
            x["pass"] for x in agreement.values()
        )
        repeats.append({"pass": passed, "oracle": comparisons, "aiter": agreement, "guards": guards})
    entry["candidate"] = {"pass": all(x["pass"] for x in repeats), "repeats": repeats, "workspace_bytes": need}
    return entry, (native, launch, inputs, native_out, candidate_out, workspace, scratch, wanted)


def quiet_gpu():
    report = _command("rocm-smi", "--showpids")
    if "KFD process information" not in report:
        raise RuntimeError("GPU PID probe unavailable")
    foreign = [pid for pid in _active_gpu_pids(report) if pid != os.getpid()]
    if foreign:
        raise RuntimeError(f"foreign active GPU processes: {foreign}")


def summarize_samples(samples, graph_repetitions):
    repeats = len(samples["aiter_ms"])
    if repeats != len(samples["candidate_ms"]) or repeats < 3:
        raise ValueError("paired latency samples required")
    aiter_median = statistics.median(samples["aiter_ms"])
    candidate_median = statistics.median(samples["candidate_ms"])
    ordered_aiter = sorted(samples["aiter_ms"])
    ordered_candidate = sorted(samples["candidate_ms"])
    p95_index = max(0, int(0.95 * repeats + 0.999999) - 1)
    baseline_cv = statistics.pstdev(samples["aiter_ms"]) / aiter_median
    baseline_mad = statistics.median(abs(value - aiter_median) for value in samples["aiter_ms"]) / aiter_median
    ratio = candidate_median / aiter_median
    return {
        **samples, "method": "graph_replay" if graph_repetitions else "python_event",
        "graph_repetitions": graph_repetitions,
        "aiter_median_ms": aiter_median,
        "candidate_median_ms": candidate_median,
        "aiter_p95_ms": ordered_aiter[p95_index],
        "candidate_p95_ms": ordered_candidate[p95_index],
        "baseline_cv": baseline_cv, "baseline_relative_mad": baseline_mad,
        "median_ratio": ratio,
        "noise_qualified": graph_repetitions > 0 and baseline_mad <= 0.05,
        "exploratory_parity": graph_repetitions > 0 and baseline_mad <= 0.05 and ratio <= 1.05,
    }


def benchmark(native, candidate, guard_fn, warmup, repeats, graph_repetitions=0):
    import torch

    quiet_gpu()
    for _ in range(warmup):
        native()
        candidate()
    torch.cuda.synchronize()
    if graph_repetitions:
        native_graph = torch.cuda.CUDAGraph()
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(native_graph):
            for _ in range(graph_repetitions):
                native()
        with torch.cuda.graph(candidate_graph):
            for _ in range(graph_repetitions):
                candidate()
        torch.cuda.synchronize()
        native_call, candidate_call = native_graph.replay, candidate_graph.replay
    else:
        native_call, candidate_call = native, candidate
    samples = {"aiter_ms": [], "candidate_ms": []}
    for index in range(repeats):
        order = (("aiter_ms", native_call), ("candidate_ms", candidate_call))
        if index % 2:
            order = tuple(reversed(order))
        for key, fn in order:
            samples[key].append(_timed_call(fn) / max(1, graph_repetitions))
    torch.cuda.synchronize()
    quiet_gpu()
    if not guard_fn():
        raise RuntimeError("guard or input mutation after benchmark")
    return summarize_samples(samples, graph_repetitions)


def score(args):
    import torch

    spec = json.loads(args.spec.read_text())
    revision = subprocess.check_output(["git", "-C", str(args.aiter_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["aiter_sha"]:
        raise RuntimeError("AITER revision mismatch")
    sys.path.insert(0, str(args.aiter_source))
    module = importlib.import_module("aiter.ops.mhc")
    environment = _gpu_manifest(spec, args.aiter_source, args.host_gpu_report)
    if module.get_gfx_runtime() != "gfx950" or module.get_cu_num() != 256:
        raise RuntimeError("MHC runtime gfx950/256-CU policy mismatch")
    candidate = Candidate(args.candidate)
    selected = set(args.case or [case["id"] for case in spec["cases"]])
    if selected - {case["id"] for case in spec["cases"]}:
        raise ValueError("unknown case")
    entries = []
    for case in spec["cases"]:
        if case["id"] not in selected:
            continue
        try:
            entry, funcs = run_case(module, candidate, case)
            if args.benchmark and entry["aiter"]["pass"] and entry["candidate"]["pass"]:
                native, launch, inputs, native_out, candidate_out, workspace, scratch, wanted = funcs
                guard_fn = lambda: checks(inputs, native_out, scratch=scratch)["pass"] and checks(inputs, candidate_out, workspace)["pass"]
                entry["timing"] = benchmark(native, launch, guard_fn, args.warmup, args.repeats, args.graph_repetitions)
                entry["timing"]["post_benchmark_aiter"] = compare_all(wanted, native_out)
                entry["timing"]["post_benchmark_candidate"] = compare_all(wanted, candidate_out)
                if not all(x["pass"] for group in (entry["timing"]["post_benchmark_aiter"], entry["timing"]["post_benchmark_candidate"]) for x in group.values()):
                    entry["timing"]["exploratory_parity"] = False
                    entry["timing"]["invalid_output_after_repeats"] = True
            entries.append(entry)
        except Exception as exc:
            entries.append({"id": case["id"], "pass": False, "error": repr(exc), "traceback": traceback.format_exc()})
    complete = len(entries) == len(spec["cases"])
    all_pass = all(item.get("aiter", {}).get("pass") and item.get("candidate", {}).get("pass") for item in entries)
    return {
        "schema_version": 1, "task_id": spec["task_id"], "candidate_kind": "analyst_unscored",
        "aiter_sha": revision, "spec_raw_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "environment": environment, "correctness": {
            "status": ("complete_pass" if complete else "partial_pass") if all_pass else "fail",
            "complete_matrix": complete, "case_count": len(entries), "cases": entries,
        },
        "performance": {"status": "exploratory" if any("timing" in item for item in entries) else "not_run"},
        "scored_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "candidate", "aiter_source", "host_gpu_report", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--graph-repetitions", type=int, default=0,
                        help="capture each exact call boundary N times per graph replay")
    parser.add_argument("--unscored-analyst", action="store_true", required=True)
    args = parser.parse_args()
    if not args.unscored_analyst or args.warmup < 0 or not 3 <= args.repeats <= 100 or args.graph_repetitions < 0 or args.graph_repetitions > 128:
        parser.error("analyst-only pilot with 3-100 timing repeats and at most 128 graph repetitions")
    for name in ("spec", "candidate", "aiter_source", "host_gpu_report", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.is_relative_to(Path.cwd().resolve()):
        parser.error("raw output must be outside repository")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = score(args)
    except Exception as exc:
        result = {"schema_version": 1, "error": repr(exc), "traceback": traceback.format_exc(),
                  "correctness": {"status": "setup_error"}, "performance": {"status": "not_run"},
                  "scored_eligible": False}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = args.output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    print(json.dumps({"correctness": result["correctness"]["status"], "case_count": result["correctness"].get("case_count"), "result": str(path)}))
    return 0 if result["correctness"]["status"] in ("partial_pass", "complete_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
