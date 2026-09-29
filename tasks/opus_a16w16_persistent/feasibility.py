"""Unscored, paired graph feasibility of pinned OPUS and its source adapter."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import sys
from pathlib import Path

from admit import _guard_intact, _guarded_tensor, checked_aiter_sha, validate_spec

GRAPH_CALLS = 32
REPEATS = 20


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_adapter(path: Path):
    library = ctypes.CDLL(str(path))
    run = library.aiter_rs_opus_a16w16_persistent
    run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    run.restype = ctypes.c_int
    return run


def _compare(torch, output, reference, atol: float, rtol: float) -> dict:
    actual = output.float()
    error = (actual - reference).abs()
    bad = (~torch.isfinite(actual)) | (error > atol + rtol * reference.abs())
    count = int(bad.sum().item())
    return {
        "pass": count == 0,
        "bad_count": count,
        "max_abs_error": float(error.max().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _capture(torch, call):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_CALLS):
            call()
    return graph


def _timed_replay(torch, graph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GRAPH_CALLS


def summarize(samples: dict[str, list[float]]) -> dict:
    baseline = samples["aiter_us"]
    candidate = samples["adapter_us"]
    if len(baseline) != REPEATS or len(candidate) != REPEATS or min(baseline + candidate) <= 0:
        raise ValueError("paired positive graph samples required")
    base_med = statistics.median(baseline)
    cand_med = statistics.median(candidate)
    base_mad = statistics.median(abs(v - base_med) for v in baseline) / base_med
    cand_mad = statistics.median(abs(v - cand_med) for v in candidate) / cand_med
    return {
        "method": "32_call_graph_replay_gpu_events_paired_alternating",
        "samples": samples,
        "aiter_median_us_per_call": base_med,
        "adapter_median_us_per_call": cand_med,
        "adapter_to_aiter_median_ratio": cand_med / base_med,
        "aiter_relative_mad": base_mad,
        "adapter_relative_mad": cand_mad,
        "noise_qualified": max(base_mad, cand_mad) <= 0.05,
    }


def run(spec: dict, adapter_path: Path) -> dict:
    import torch
    from aiter.ops.opus import opus_bmm

    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"expected gfx950, got {arch!r}")
    adapter = _load_adapter(adapter_path)
    result = {"schema": "aiter-rs-opus-a16w16-source-adapter-feasibility-v1", "gpu_arch": arch, "cases": []}
    for case in spec["cases"]:
        m, n, k, kid = (case[key] for key in ("m", "n", "k", "kid"))
        torch.manual_seed(case["seed"])
        a_storage, a, guard = _guarded_tensor(torch, (1, m, k), sentinel=13.0)
        b_storage, b, _ = _guarded_tensor(torch, (1, n, k), sentinel=-17.0)
        base_storage, baseline, _ = _guarded_tensor(torch, (1, m, n), sentinel=23.0)
        cand_storage, candidate, _ = _guarded_tensor(torch, (1, m, n), sentinel=29.0)
        a.copy_(torch.randn(a.shape, device="cuda", dtype=torch.float32))
        b.copy_(torch.randn(b.shape, device="cuda", dtype=torch.float32))
        original_a, original_b = a.clone(), b.clone()
        reference = torch.matmul(a.float(), b.float().transpose(-1, -2))

        def base_call():
            opus_bmm(a, b, baseline, kid=kid, split_k=0)

        def candidate_call():
            stream = torch.cuda.current_stream().cuda_stream
            status = adapter(a.data_ptr(), b.data_ptr(), candidate.data_ptr(), m, n, k, kid, stream)
            if status:
                raise RuntimeError(f"source adapter returned hipError_t={status}")

        checks = []
        for repeat in range(2):
            baseline.fill_(float("nan"))
            candidate.fill_(float("nan"))
            base_call()
            candidate_call()
            torch.cuda.synchronize()
            base_check = _compare(torch, baseline, reference, spec["atol"], spec["rtol"])
            cand_check = _compare(torch, candidate, reference, spec["atol"], spec["rtol"])
            identical = bool(torch.equal(baseline, candidate))
            guards = all((
                _guard_intact(torch, a_storage, guard, 13.0),
                _guard_intact(torch, b_storage, guard, -17.0),
                _guard_intact(torch, base_storage, guard, 23.0),
                _guard_intact(torch, cand_storage, guard, 29.0),
            ))
            inputs_unchanged = bool(torch.equal(a, original_a) and torch.equal(b, original_b))
            checks.append({
                "repeat": repeat,
                "aiter": base_check,
                "adapter": cand_check,
                "bitwise_same_output": identical,
                "guards_intact": guards,
                "inputs_unchanged": inputs_unchanged,
                "pass": base_check["pass"] and cand_check["pass"] and guards and inputs_unchanged,
            })
        entry = {"id": case["id"], "kid": kid, "correctness": checks}
        if all(check["pass"] for check in checks):
            for _ in range(5):
                base_call()
                candidate_call()
            torch.cuda.synchronize()
            base_graph = _capture(torch, base_call)
            cand_graph = _capture(torch, candidate_call)
            torch.cuda.synchronize()
            for _ in range(2):
                base_graph.replay()
                cand_graph.replay()
            torch.cuda.synchronize()
            samples = {"aiter_us": [], "adapter_us": []}
            for index in range(REPEATS):
                order = [("aiter_us", base_graph), ("adapter_us", cand_graph)]
                if index % 2:
                    order.reverse()
                for key, graph in order:
                    samples[key].append(_timed_replay(torch, graph))
            entry["graph_performance"] = summarize(samples)
            entry["post_graph"] = {
                "aiter": _compare(torch, baseline, reference, spec["atol"], spec["rtol"]),
                "adapter": _compare(torch, candidate, reference, spec["atol"], spec["rtol"]),
                "bitwise_same_output": bool(torch.equal(baseline, candidate)),
                "guards_intact": all((
                    _guard_intact(torch, a_storage, guard, 13.0),
                    _guard_intact(torch, b_storage, guard, -17.0),
                    _guard_intact(torch, base_storage, guard, 23.0),
                    _guard_intact(torch, cand_storage, guard, 29.0),
                )),
                "inputs_unchanged": bool(torch.equal(a, original_a) and torch.equal(b, original_b)),
            }
        result["cases"].append(entry)
    result["all_correct"] = all(
        all(check["pass"] for check in entry["correctness"])
        and entry.get("post_graph", {}).get("aiter", {}).get("pass", False)
        and entry.get("post_graph", {}).get("adapter", {}).get("pass", False)
        and entry.get("post_graph", {}).get("guards_intact", False)
        and entry.get("post_graph", {}).get("inputs_unchanged", False)
        for entry in result["cases"]
    )
    result["all_noise_qualified"] = all(
        entry.get("graph_performance", {}).get("noise_qualified", False) for entry in result["cases"]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--aiter-source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_path, source, adapter_path, output = (
        path.resolve() for path in (args.spec, args.aiter_source, args.adapter, args.output)
    )
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("output must be a new off-source file")
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    checked_aiter_sha(source, spec["aiter_sha"])
    sys.path.insert(0, str(source))
    result = run(spec, adapter_path)
    result.update({
        "aiter_sha": spec["aiter_sha"],
        "spec_sha256": _sha(spec_path),
        "adapter_sha256": _sha(adapter_path),
        "feasibility_script_sha256": _sha(Path(__file__)),
        "scored": False,
        "agent_candidate": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
