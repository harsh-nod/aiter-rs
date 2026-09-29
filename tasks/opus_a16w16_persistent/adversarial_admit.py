"""One-case, correctness-only public OPUS admission; run under a host watchdog."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.run import _active_gpu_pids
from tasks.opus_a16w16_persistent.admit import (
    _guard_intact,
    _guarded_tensor,
    _profile_kernel_names,
    checked_aiter_sha,
    sha256_file,
)
from tasks.opus_a16w16_persistent.adversarial_matrix import (
    AITERSHA,
    compare_output,
    fp32_reference,
    load_matrix,
    make_operands,
)

MATRIX_SHA = "e4a2346d96db18d6f071fae8b1caeffd327342dcc6976c762d47af37159fd3ed"
ADAPTER_SOURCE_SHA = "56556edb068aa1b761b23d554b4b54223c627de69710ce04da38beeffea5fa36"
ADAPTER_BINARY_SHA = "356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50"
HEADER_SHA = "bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b"
GUARD = 512


def select_public_case(matrix: dict, case_id: str) -> dict:
    matches = [case for case in matrix["cases"] if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case is not in the proposed public matrix")
    return matches[0]


def check_provenance(args) -> dict:
    matrix = load_matrix(args.matrix, args.anchors)
    if sha256_file(args.matrix) != MATRIX_SHA:
        raise ValueError("proposed public matrix raw hash changed")
    if sha256_file(args.adapter_source) != ADAPTER_SOURCE_SHA:
        raise ValueError("standalone adapter source raw hash changed")
    if sha256_file(args.adapter) != ADAPTER_BINARY_SHA:
        raise ValueError("standalone adapter binary raw hash changed")
    checked_aiter_sha(args.aiter_source, AITERSHA)
    header = args.aiter_source / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh"
    if sha256_file(header) != HEADER_SHA:
        raise ValueError("pinned persistent header changed")
    return matrix


def _gpu_pids() -> list[int]:
    output = subprocess.check_output(["rocm-smi", "--showpids"], text=True, timeout=15)
    if "KFD process information" not in output:
        raise RuntimeError("GPU PID preflight unavailable")
    return _active_gpu_pids(output)


def _gpu_attestation(torch, host_report: str) -> dict:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU required")
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", ""))
    product = subprocess.check_output(["rocm-smi", "--showid"], text=True, timeout=15)
    name = _attest_sku(str(props.name), arch, product, host_report)
    return {"gpu_name": name, "gpu_arch": arch, "gpu_pci_id": "0x75a0", "torch": torch.__version__, "hip": torch.version.hip}


def _device_guid(report: str) -> str | None:
    match = re.search(r"GPU\[0\]\s*:\s*GUID:\s*(\d+)", report)
    return match.group(1) if match else None


def _attest_sku(torch_name: str, arch: str, container_product: str, host_report: str) -> str:
    host_guid, container_guid = _device_guid(host_report), _device_guid(container_product)
    if (
        not arch.startswith("gfx950")
        or "Device ID:" not in host_report or "0x75a0" not in host_report
        or "AMD Instinct MI350X" not in host_report
        or "Device ID:" not in container_product or "0x75a0" not in container_product
        or host_guid is None or host_guid != container_guid
    ):
        raise RuntimeError("host/container MI350X/gfx950/PCI/GUID not jointly attested")
    if torch_name and "MI350X" not in torch_name:
        raise RuntimeError("Torch GPU name contradicts ROCm product")
    return torch_name or "AMD Instinct MI350X"


def _adapter_call(adapter, a, b, y, case, stream) -> None:
    status = adapter(a.data_ptr(), b.data_ptr(), y.data_ptr(), case["m"], case["n"], case["k"], case["kid"], stream)
    if status:
        raise RuntimeError(f"standalone HIP adapter returned hipError_t={status}")


def _load_adapter(path: Path):
    run = ctypes.CDLL(str(path)).aiter_rs_opus_a16w16_persistent
    run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    run.restype = ctypes.c_int
    return run


def _clean_check(check: dict) -> dict:
    if not math.isfinite(check["max_abs_error"]):
        check["max_abs_error"] = None
    return check


def _bitwise_equal(torch, left, right) -> bool:
    return bool(torch.equal(left.view(torch.int16), right.view(torch.int16)))


def _dispatch_matches(names: list[str], kid: int) -> bool:
    gemm = [name for name in names if "gemm_a16w16" in name]
    specialization = "ELb1" if kid == 300 else "ELb0"
    return len(gemm) == 1 and "gemm_a16w16_persistent_kernel" in gemm[0] and specialization in gemm[0]


def run_case(case: dict, adapter_path: Path, host_report: str) -> dict:
    import torch
    from aiter.ops.opus import opus_bmm
    from aiter.ops.opus._arch import _device_arch_and_cu
    from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
    from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    env = _gpu_attestation(torch, host_report)
    active = _gpu_pids()
    if any(pid != os.getpid() for pid in active):
        raise RuntimeError("foreign active GPU process at preflight")

    kid = case["kid"]
    instance = get_kernel_instance("gfx950", "a16w16", kid, torch.bfloat16)
    if instance is None or instance.kernel_tag != "a16w16_persistent":
        raise RuntimeError("kid does not resolve to gfx950 persistent instance")
    arch, cu_count = _device_arch_and_cu(torch.device("cuda:0"))
    plan = _get_cached_a16w16_launch_plan(
        arch, case["m"], case["n"], case["k"], 1, cu_count,
        False, torch.bfloat16, torch.bfloat16, kid, 0,
    )
    if plan.resolved_kid != kid or plan.workspace_spec is not None:
        raise RuntimeError("kid switched or workspace route used")
    adapter = _load_adapter(adapter_path)
    m, n, k = (case[key] for key in ("m", "n", "k"))
    a_store, a, _ = _guarded_tensor(torch, (1, m, k), sentinel=13.0)
    b_store, b, _ = _guarded_tensor(torch, (1, n, k), sentinel=-17.0)
    base_store, base, _ = _guarded_tensor(torch, (1, m, n), sentinel=23.0)
    cand_store, cand, _ = _guarded_tensor(torch, (1, m, n), sentinel=29.0)
    base_alt_store, base_alt, _ = _guarded_tensor(torch, (1, m, n), sentinel=31.0)
    cand_alt_store, cand_alt, _ = _guarded_tensor(torch, (1, m, n), sentinel=37.0)
    storages = (
        (a_store, 13.0), (b_store, -17.0), (base_store, 23.0),
        (cand_store, 29.0), (base_alt_store, 31.0), (cand_alt_store, 37.0),
    )
    stream = torch.cuda.current_stream().cuda_stream
    steps = []
    phase0_outputs = None
    active_output = (base, cand)

    for label, phase, use_alternate in (
        ("phase0_first", 0, False),
        ("phase0_repeat", 0, False),
        ("phase1_reuse", 1, False),
        ("phase1_alternate_output", 1, True),
        ("phase0_restore", 0, False),
        ("phase0_guard_perturbation", 0, False),
    ):
        expected_a, expected_b = make_operands(torch, case, device="cuda", phase=phase)
        a.copy_(expected_a)
        b.copy_(expected_b)
        reference = fp32_reference(torch, a, b)
        torch.cuda.synchronize()
        if label == "phase0_first":
            base.fill_(float("nan"))
            cand.fill_(float("nan"))
        if label == "phase0_guard_perturbation":
            a_store[:GUARD].fill_(41.0)
            a_store[-GUARD:].fill_(41.0)
            b_store[:GUARD].fill_(-43.0)
            b_store[-GUARD:].fill_(-43.0)
        inactive = tuple(output.clone() for output in active_output) if use_alternate else None
        y_base, y_cand = (base_alt, cand_alt) if use_alternate else active_output
        opus_bmm(a, b, y_base, kid=kid, split_k=0)
        torch.cuda.synchronize()
        baseline_check = _clean_check(compare_output(torch, y_base, reference))
        _adapter_call(adapter, a, b, y_cand, case, stream)
        torch.cuda.synchronize()
        candidate_check = _clean_check(compare_output(torch, y_cand, reference))
        guard_ok = all(_guard_intact(torch, storage, GUARD, sentinel) for storage, sentinel in storages[2:])
        if label == "phase0_guard_perturbation":
            input_guard_ok = all((
                _guard_intact(torch, a_store, GUARD, 41.0),
                _guard_intact(torch, b_store, GUARD, -43.0),
            ))
        else:
            input_guard_ok = all((
                _guard_intact(torch, a_store, GUARD, 13.0),
                _guard_intact(torch, b_store, GUARD, -17.0),
            ))
        input_ok = _bitwise_equal(torch, a, expected_a) and _bitwise_equal(torch, b, expected_b)
        bitwise_same = _bitwise_equal(torch, y_base, y_cand)
        inactive_ok = inactive is None or all(
            _bitwise_equal(torch, before, after) for before, after in zip(inactive, active_output)
        )
        if label == "phase0_first":
            phase0_outputs = (y_base.clone(), y_cand.clone())
        phase0_reproduced = None
        if label in ("phase0_repeat", "phase0_restore", "phase0_guard_perturbation"):
            phase0_reproduced = all(
                _bitwise_equal(torch, before, after) for before, after in zip(phase0_outputs, (y_base, y_cand))
            )
        step = {
            "label": label,
            "baseline": baseline_check,
            "adapter": candidate_check,
            "bitwise_same": bitwise_same,
            "input_unchanged": input_ok,
            "guards_intact": guard_ok and input_guard_ok,
            "inactive_output_unchanged": inactive_ok,
            "phase0_reproduced": phase0_reproduced,
        }
        step["pass"] = all((
            baseline_check["pass"], candidate_check["pass"], input_ok,
            guard_ok, input_guard_ok, inactive_ok,
            phase0_reproduced is not False,
        ))
        steps.append(step)
        if not step["pass"]:
            break

    dispatch = {"checked": False, "aiter": [], "adapter": [], "post_profile_checks": None}
    if all(step["pass"] for step in steps) and len(steps) == 6:
        a_store[:GUARD].fill_(13.0)
        a_store[-GUARD:].fill_(13.0)
        b_store[:GUARD].fill_(-17.0)
        b_store[-GUARD:].fill_(-17.0)
        dispatch["aiter"] = _profile_kernel_names(torch, lambda: opus_bmm(a, b, base, kid=kid, split_k=0))
        dispatch["adapter"] = _profile_kernel_names(torch, lambda: _adapter_call(adapter, a, b, cand, case, stream))
        profile_guards = all(_guard_intact(torch, storage, GUARD, sentinel) for storage, sentinel in storages)
        post_profile = {
            "aiter": _clean_check(compare_output(torch, base, reference)),
            "adapter": _clean_check(compare_output(torch, cand, reference)),
            "guards_intact": profile_guards,
            "inputs_unchanged": _bitwise_equal(torch, a, expected_a) and _bitwise_equal(torch, b, expected_b),
        }
        dispatch["post_profile_checks"] = post_profile
        dispatch["checked"] = (
            all(_dispatch_matches(dispatch[path], kid) for path in ("aiter", "adapter"))
            and post_profile["aiter"]["pass"] and post_profile["adapter"]["pass"]
            and profile_guards and post_profile["inputs_unchanged"]
        )
    post_pids = _gpu_pids()
    pid_ok = os.getpid() in post_pids and not any(pid != os.getpid() for pid in post_pids)
    return {
        "case_id": case["id"], "kid": kid, "source": case["source"],
        "environment": env, "resolved_kid": plan.resolved_kid,
        "workspace_used": plan.workspace_spec is not None,
        "steps": steps, "dispatch": dispatch,
        "host_pid_attributed": pid_ok,
        "pass": len(steps) == 6 and all(step["pass"] for step in steps) and dispatch["checked"] and pid_ok,
        "performance": "not_run", "withheld_cases_evaluated": False,
        "scored_agent_candidate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("matrix", "anchors", "aiter-source", "adapter-source", "adapter", "host-gpu-report", "case", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path if field != "case" else str)
    args = parser.parse_args()
    for field in ("matrix", "anchors", "aiter_source", "adapter_source", "adapter", "host_gpu_report", "output"):
        setattr(args, field, getattr(args, field).resolve())
    repository = Path(__file__).resolve().parents[2]
    if args.output.exists() or args.output.is_relative_to(repository) or args.output.is_relative_to(args.aiter_source):
        parser.error("output must be a new private file outside source trees")
    matrix = check_provenance(args)
    case = select_public_case(matrix, args.case)
    if _gpu_pids():
        raise RuntimeError("foreign GPU process at host preflight")
    sys.path.insert(0, str(args.aiter_source))
    result = run_case(case, args.adapter, args.host_gpu_report.read_text(encoding="utf-8"))
    result.update({
        "schema": "aiter-rs-opus-adversarial-public-one-case-v1",
        "aiter_sha": AITERSHA,
        "matrix_sha256": MATRIX_SHA,
        "adapter_source_sha256": ADAPTER_SOURCE_SHA,
        "adapter_binary_sha256": ADAPTER_BINARY_SHA,
        "driver_sha256": sha256_file(Path(__file__)),
        "host_gpu_report_sha256": sha256_file(args.host_gpu_report),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    args.output.chmod(0o444)
    print(json.dumps({"case": args.case, "pass": result["pass"], "steps": len(result["steps"])}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
