"""Correctness-only admission probe for exact-kid gfx950 OPUS persistent GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

NUM_CU = 256
NUM_XCD = 8
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
PERSISTENT_KIDS = {300, 1300}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch_geometry(m: int, n: int) -> dict[str, int]:
    """Mirror the pinned gfx950 persistent host grid calculation."""
    tiles_m = (m + BLOCK_M - 1) // BLOCK_M
    tiles_n = (n + BLOCK_N - 1) // BLOCK_N
    split_m = max(1, (NUM_CU + tiles_n - 1) // tiles_n)
    while split_m < tiles_m and tiles_m % split_m:
        split_m += 1
    split_m = min(split_m, tiles_m)
    if tiles_m % split_m:
        raise ValueError("persistent grid must divide tile M count")
    groups_per_xcd = (split_m + NUM_XCD - 1) // NUM_XCD
    return {
        "tiles_m": tiles_m,
        "tiles_n": tiles_n,
        "split_m": split_m,
        "m_per_wg": tiles_m // split_m,
        "grid_y_padded": groups_per_xcd * NUM_XCD,
    }


def validate_spec(spec: dict) -> None:
    if spec.get("schema") != "aiter-rs-opus-a16w16-persistent-admission-v1":
        raise ValueError("unknown spec schema")
    if spec.get("arch") != "gfx950" or spec.get("dtype") != "bfloat16" or spec.get("output_dtype") != "bfloat16":
        raise ValueError("this probe only covers gfx950 BF16 -> BF16")
    if spec.get("repeats") != 2 or spec.get("rtol") != 0.02 or spec.get("atol") != 0.125:
        raise ValueError("admission tolerances/repeats are frozen")
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("expected exactly three frozen cases")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("case IDs must be distinct")
    for case in cases:
        if case["batch"] != 1 or case["kid"] not in PERSISTENT_KIDS:
            raise ValueError("unsupported batch or kid")
        m, n, k = (case[key] for key in ("m", "n", "k"))
        loops = (k + BLOCK_K - 1) // BLOCK_K
        if min(m, n, k) < 1 or k % 2 or loops < 2 or loops % 2:
            raise ValueError("invalid persistent shape/K-loop")
        if n % 16:
            raise ValueError("persistent tuner excludes N not divisible by 16")
        if case["kid"] == 1300 and (m % BLOCK_M or n % BLOCK_N or k % BLOCK_K):
            raise ValueError("no-OOB kid requires full M/N/K tiles")
        if launch_geometry(m, n)["m_per_wg"] < 2:
            raise ValueError("case does not actually iterate persistent M tiles")


def checked_aiter_sha(source: Path, expected: str) -> str:
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if sha != expected:
        raise ValueError(f"AITER SHA mismatch: {sha}")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise ValueError("AITER checkout is not clean")
    return sha


def _guarded_tensor(torch, shape: tuple[int, ...], *, sentinel: float):
    count = 1
    for dim in shape:
        count *= dim
    guard = 512
    storage = torch.full((count + 2 * guard,), sentinel, dtype=torch.bfloat16, device="cuda")
    value = storage[guard : guard + count].view(shape)
    return storage, value, guard


def _guard_intact(torch, storage, guard: int, sentinel: float) -> bool:
    return bool(torch.all(storage[:guard] == sentinel).item() and torch.all(storage[-guard:] == sentinel).item())


def _profile_kernel_names(torch, call) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names = sorted({
        event.name
        for event in prof.events()
        if str(event.device_type).endswith("CUDA")
    })
    return names


def run_probe(spec: dict, *, profile_dispatch: bool) -> dict:
    import torch
    from aiter.ops.opus import opus_bmm
    from aiter.ops.opus._arch import _device_arch_and_cu
    from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
    from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    arch_name = str(getattr(props, "gcnArchName", ""))
    if not arch_name.startswith("gfx950"):
        raise RuntimeError(f"expected gfx950, got {arch_name!r}")

    result = {
        "schema": "aiter-rs-opus-a16w16-persistent-admission-result-v1",
        "gpu_name": props.name,
        "gpu_arch": arch_name,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cases": [],
        "dispatch_profiled": False,
        "dispatch_by_kid": {},
    }
    for case in spec["cases"]:
        kid = case["kid"]
        instance = get_kernel_instance("gfx950", "a16w16", kid, torch.bfloat16)
        if instance is None or instance.kernel_tag != "a16w16_persistent":
            raise RuntimeError(f"kid {kid} does not resolve to gfx950 persistent A16W16")
        arch, cu_count = _device_arch_and_cu(torch.device("cuda:0"))
        plan = _get_cached_a16w16_launch_plan(
            arch, case["m"], case["n"], case["k"], case["batch"], cu_count,
            False, torch.bfloat16, torch.bfloat16, kid, 0,
        )
        if plan.resolved_kid != kid or plan.workspace_spec is not None:
            raise RuntimeError(f"kid {kid} unexpectedly resolves to {plan.resolved_kid} or a workspace path")
        shape_a = (case["batch"], case["m"], case["k"])
        shape_b = (case["batch"], case["n"], case["k"])
        shape_y = (case["batch"], case["m"], case["n"])
        torch.manual_seed(case["seed"])
        a_storage, a, guard = _guarded_tensor(torch, shape_a, sentinel=13.0)
        b_storage, b, _ = _guarded_tensor(torch, shape_b, sentinel=-17.0)
        y_storage, y, _ = _guarded_tensor(torch, shape_y, sentinel=23.0)
        a.copy_(torch.randn(shape_a, device="cuda", dtype=torch.float32))
        b.copy_(torch.randn(shape_b, device="cuda", dtype=torch.float32))
        original_a, original_b = a.clone(), b.clone()
        with torch.no_grad():
            reference = torch.matmul(a.float(), b.float().transpose(-1, -2))
        case_result = {"id": case["id"], "kid": kid, "resolved_kid": plan.resolved_kid, "geometry": launch_geometry(case["m"], case["n"]), "repeats": []}

        def call():
            opus_bmm(a, b, y, kid=kid, split_k=0)

        for repeat in range(spec["repeats"]):
            y.fill_(float("nan"))
            call()
            torch.cuda.synchronize()
            actual = y.float()
            difference = (actual - reference).abs()
            allowance = spec["atol"] + spec["rtol"] * reference.abs()
            bad = (~torch.isfinite(actual)) | (difference > allowance)
            guard_ok = all((
                _guard_intact(torch, a_storage, guard, 13.0),
                _guard_intact(torch, b_storage, guard, -17.0),
                _guard_intact(torch, y_storage, guard, 23.0),
            ))
            inputs_ok = bool(torch.equal(a, original_a) and torch.equal(b, original_b))
            bad_count = int(bad.sum().item())
            case_result["repeats"].append({
                "repeat": repeat,
                "pass": bad_count == 0 and guard_ok and inputs_ok,
                "bad_count": bad_count,
                "max_abs_error": float(difference.max().item()),
                "output_finite": bool(torch.isfinite(actual).all().item()),
                "guards_intact": guard_ok,
                "inputs_unchanged": inputs_ok,
            })
        result["cases"].append(case_result)
        if profile_dispatch and str(kid) not in result["dispatch_by_kid"]:
            names = _profile_kernel_names(torch, call)
            matching = [name for name in names if "gemm_a16w16" in name]
            result["dispatch_by_kid"][str(kid)] = {
                "persistent_kernel_seen": any("gemm_a16w16_persistent_kernel" in name for name in matching),
                "kernel_names": matching,
            }
    result["dispatch_profiled"] = set(result["dispatch_by_kid"]) == {"300", "1300"} and all(
        route["persistent_kernel_seen"] for route in result["dispatch_by_kid"].values()
    )
    result["correctness_pass"] = all(
        repeat["pass"] for case in result["cases"] for repeat in case["repeats"]
    )
    result["admission_pass"] = result["correctness_pass"] and result["dispatch_profiled"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--aiter-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside repo/source")
    parser.add_argument("--no-profile", action="store_true", help="Development only; cannot pass admission")
    args = parser.parse_args(argv)
    spec_path, source, output = (p.resolve() for p in (args.spec, args.aiter_source, args.output))
    probe_dir = Path(__file__).resolve().parent
    repo_root = probe_dir.parents[1] if probe_dir.name == "opus_a16w16_persistent" and probe_dir.parent.name == "tasks" else probe_dir
    if output.exists() or output.is_relative_to(repo_root) or output.is_relative_to(source):
        raise ValueError("output must be a new off-repository file")
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    checked_aiter_sha(source, spec["aiter_sha"])
    sys.path.insert(0, str(source))
    result = run_probe(spec, profile_dispatch=not args.no_profile)
    result.update({
        "aiter_sha": spec["aiter_sha"],
        "spec_sha256": sha256_file(spec_path),
        "probe_sha256": sha256_file(Path(__file__)),
        "scored": False,
        "performance": "not_run",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if result["admission_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
