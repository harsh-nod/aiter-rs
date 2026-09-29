"""Trusted one-case production OPUS overlay control with isolated JIT workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tasks.opus_a16w16_persistent.scored_candidate.freeze import (
    BASE_HEADER_SHA256, HEADER, HERE, REPO,
    sha256, validate_matrix, validate_task,
)

GRAPH_CALLS = 32
REPEATS = 20
GUARD = 512
MAX_FRAME = 1024 * 1024


class Channel:
    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send(self, item: dict) -> None:
        data = json.dumps(item, allow_nan=False, sort_keys=True).encode()
        if len(data) > MAX_FRAME:
            raise ValueError("control frame too large")
        self.sock.sendall(struct.pack("!I", len(data)) + data)

    def recv(self) -> dict:
        size = struct.unpack("!I", self._read(4))[0]
        if size > MAX_FRAME:
            raise ValueError("control frame too large")
        item = json.loads(self._read(size))
        if not isinstance(item, dict):
            raise ValueError("control frame must be an object")
        return item

    def _read(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            part = self.sock.recv(size - len(chunks))
            if not part:
                raise EOFError("worker channel closed")
            chunks.extend(part)
        return bytes(chunks)


def git_changes(tree: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=all", "-z"]
    )
    entries = output.split(b"\x00")
    if entries[-1] != b"":
        raise ValueError("malformed git status")
    changed = []
    for entry in entries[:-1]:
        if len(entry) < 4 or entry[2:3] != b" " or b"R" in entry[:2] or b"C" in entry[:2]:
            raise ValueError("rename/copy or malformed git status not admitted")
        changed.append(entry[3:].decode())
    return changed


def validate_sources(baseline: Path, overlay: Path, candidate_sha: str) -> dict:
    from tasks.opus_a16w16_persistent.admit import checked_aiter_sha

    roots = [baseline.resolve(), overlay.resolve()]
    if roots[0] == roots[1] or any(root.is_relative_to(other) for root, other in ((roots[0], roots[1]), (roots[1], roots[0]))):
        raise ValueError("AITER trees must be separate")
    if len(candidate_sha) != 64 or any(c not in "0123456789abcdef" for c in candidate_sha):
        raise ValueError("candidate header hash must be SHA256")
    checked_aiter_sha(roots[0], "868ccf62a0bcad3aa47f92728340ccb37ed4fb39")
    overlay_sha = subprocess.check_output(["git", "-C", str(roots[1]), "rev-parse", "HEAD"], text=True).strip()
    if overlay_sha != "868ccf62a0bcad3aa47f92728340ccb37ed4fb39":
        raise ValueError("candidate overlay is not based on pinned AITER")
    for root in roots:
        target = root / HEADER
        if not stat.S_ISREG(target.lstat().st_mode):
            raise ValueError("persistent header must be a regular file")
    if sha256(roots[0] / HEADER) != BASE_HEADER_SHA256 or git_changes(roots[0]):
        raise ValueError("baseline AITER tree is not pinned and clean")
    changes = git_changes(roots[1])
    if changes not in ([], [HEADER.as_posix()]) or sha256(roots[1] / HEADER) != candidate_sha:
        raise ValueError("candidate overlay changed outside its pinned header")
    return {"baseline_header_sha256": BASE_HEADER_SHA256,
            "candidate_header_sha256": candidate_sha, "overlay_changed_paths": changes}


def validate_runtime_paths(baseline: Path, overlay: Path, jit_baseline: Path, jit_candidate: Path, output: Path) -> None:
    roots = [path.resolve() for path in (baseline, overlay, jit_baseline, jit_candidate, output)]
    if len(set(roots)) != len(roots):
        raise ValueError("source, JIT, and output paths must differ")
    for index, path in enumerate(roots):
        for other in roots[index + 1:]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("source, JIT, and output paths must not nest")
    if any(path.is_relative_to(REPO) for path in roots[2:]):
        raise ValueError("JIT and raw output must stay outside the repository")
    if jit_baseline.exists() or jit_candidate.exists() or output.exists():
        raise ValueError("JIT caches and raw output must be new")
    if output.parent.stat().st_mode & 0o077:
        raise ValueError("raw output parent must be mode 700")


def read_matrix(path: Path, task: dict, *, withheld: bool) -> dict:
    field = "withheld_matrix_sha256" if withheld else "public_matrix_sha256"
    if sha256(path) != task[field]:
        raise ValueError("matrix raw commitment mismatch")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    validate_matrix(matrix, withheld=withheld)
    return matrix


def select_case(matrix: dict, case_id: str) -> dict:
    matches = [case for case in matrix["cases"] if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case ID is not in the committed matrix")
    return matches[0]


def _same_bits(torch, a, b) -> bool:
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def _clean_check(check: dict) -> dict:
    if not math.isfinite(check["max_abs_error"]):
        check["max_abs_error"] = None
    return check


def _worker_init(source: Path, jit: Path, host_report: str, case: dict, *, benchmark: bool):
    from tasks.opus_a16w16_persistent.admit import _guard_intact, _guarded_tensor, _profile_kernel_names
    from tasks.opus_a16w16_persistent.adversarial_admit import _dispatch_matches, _gpu_attestation
    from tasks.opus_a16w16_persistent.adversarial_matrix import compare_output, fp32_reference, make_operands

    os.environ["AITER_JIT_DIR"] = str(jit)
    sys.path.insert(0, str(source))
    import torch
    import aiter
    from aiter.ops.opus import opus_bmm
    from aiter.ops.opus._arch import _device_arch_and_cu
    from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
    from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

    if not Path(aiter.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("AITER Python wrapper imported outside this worker's pinned source")

    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    env = _gpu_attestation(torch, host_report)
    kid = case["kid"]
    instance = get_kernel_instance("gfx950", "a16w16", kid, torch.bfloat16)
    if instance is None or instance.kernel_tag != "a16w16_persistent":
        raise RuntimeError("requested kid is not gfx950 persistent")
    arch, cu_count = _device_arch_and_cu(torch.device("cuda:0"))
    plan = _get_cached_a16w16_launch_plan(
        arch, case["m"], case["n"], case["k"], 1, cu_count,
        False, torch.bfloat16, torch.bfloat16, kid, 0,
    )
    if plan.resolved_kid != kid or plan.workspace_spec is not None:
        raise RuntimeError("kid switched or workspace route used")

    m, n, k = (case[field] for field in ("m", "n", "k"))
    a_store, a, _ = _guarded_tensor(torch, (1, m, k), sentinel=13.0)
    b_store, b, _ = _guarded_tensor(torch, (1, n, k), sentinel=-17.0)
    y_store, y, _ = _guarded_tensor(torch, (1, m, n), sentinel=23.0)
    alt_store, alt, _ = _guarded_tensor(torch, (1, m, n), sentinel=31.0)
    storages = ((a_store, 13.0), (b_store, -17.0), (y_store, 23.0), (alt_store, 31.0))

    def call(target=y):
        opus_bmm(a, b, target, kid=kid, split_k=0)

    steps = []
    phase0_output = None
    for label, phase, use_alt in (
        ("phase0_first", 0, False), ("phase0_repeat", 0, False),
        ("phase1_reuse", 1, False), ("phase1_alternate_output", 1, True),
        ("phase0_restore", 0, False), ("phase0_guard_perturbation", 0, False),
    ):
        expected_a, expected_b = make_operands(torch, case, device="cuda", phase=phase)
        a.copy_(expected_a)
        b.copy_(expected_b)
        reference = fp32_reference(torch, a, b)
        if label == "phase0_first":
            y.fill_(float("nan"))
        if label == "phase0_guard_perturbation":
            a_store[:GUARD].fill_(41.0)
            a_store[-GUARD:].fill_(41.0)
            b_store[:GUARD].fill_(-43.0)
            b_store[-GUARD:].fill_(-43.0)
        inactive_before = y.clone() if use_alt else None
        target = alt if use_alt else y
        call(target)
        torch.cuda.synchronize()
        output = _clean_check(compare_output(torch, target, reference))
        expected_guards = ((a_store, 41.0), (b_store, -43.0)) if label == "phase0_guard_perturbation" else storages[:2]
        guard_ok = all(_guard_intact(torch, storage, GUARD, sentinel) for storage, sentinel in (*expected_guards, *storages[2:]))
        input_ok = _same_bits(torch, a, expected_a) and _same_bits(torch, b, expected_b)
        inactive_ok = inactive_before is None or _same_bits(torch, inactive_before, y)
        if label == "phase0_first":
            phase0_output = y.clone()
        reproduce = None if label not in ("phase0_repeat", "phase0_restore", "phase0_guard_perturbation") else _same_bits(torch, y, phase0_output)
        passed = all((output["pass"], guard_ok, input_ok, inactive_ok, reproduce is not False))
        steps.append({"label": label, "oracle": output, "guards_intact": guard_ok,
                      "inputs_unchanged": input_ok, "inactive_unchanged": inactive_ok,
                      "phase0_reproduced": reproduce, "pass": passed})
        if not passed:
            break
    if len(steps) != 6 or not all(step["pass"] for step in steps):
        return {"correctness_pass": False, "steps": steps, "environment": env}, None

    a_store[:GUARD].fill_(13.0)
    a_store[-GUARD:].fill_(13.0)
    b_store[:GUARD].fill_(-17.0)
    b_store[-GUARD:].fill_(-17.0)
    names = _profile_kernel_names(torch, lambda: call())
    torch.cuda.synchronize()
    profile_check = _clean_check(compare_output(torch, y, reference))
    profile_ok = (_dispatch_matches(names, kid) and profile_check["pass"]
                  and all(_guard_intact(torch, storage, GUARD, sentinel) for storage, sentinel in storages)
                  and _same_bits(torch, a, expected_a) and _same_bits(torch, b, expected_b))
    from aiter.jit.core import get_module
    module = get_module("module_deepgemm_opus")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(jit.resolve()):
        raise RuntimeError("JIT module loaded outside this worker's isolated cache")
    result = {"correctness_pass": profile_ok, "steps": steps, "exact_dispatch": profile_ok,
              "profile_kernel_names": names, "environment": env,
              "jit_module_sha256": sha256(module_path), "jit_module_path": str(module_path),
              "resolved_kid": plan.resolved_kid, "workspace_used": False}
    if not profile_ok or not benchmark:
        return result, None

    generator = torch.Generator(device="cuda").manual_seed(case["seed"] + 100_000)
    a.copy_(torch.randn(a.shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16))
    b.copy_(torch.randn(b.shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16))
    original_a, original_b = a.clone(), b.clone()
    input_hashes = {
        "a_sha256": hashlib.sha256(a.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        "b_sha256": hashlib.sha256(b.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
    }
    result["performance_input_hashes"] = input_hashes
    reference = fp32_reference(torch, a, b)
    y.fill_(float("nan"))
    call()
    torch.cuda.synchronize()

    def inspect():
        check = _clean_check(compare_output(torch, y, reference))
        guard_ok = all(_guard_intact(torch, storage, GUARD, sentinel) for storage, sentinel in storages)
        input_ok = _same_bits(torch, a, original_a) and _same_bits(torch, b, original_b)
        return {"oracle": check, "guards_intact": guard_ok, "inputs_unchanged": input_ok,
                "pass": check["pass"] and guard_ok and input_ok}

    result["pre_graph"] = inspect()
    if not result["pre_graph"]["pass"]:
        return result, None
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_CALLS):
            call()
    torch.cuda.synchronize()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    result["pre_timing"] = inspect()

    def replay():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / GRAPH_CALLS

    return result, (replay, inspect)


def worker_loop(channel: Channel, *, fake: bool = False) -> None:
    state = None
    while True:
        request = channel.recv()
        command = request.get("command")
        try:
            if command == "close":
                channel.send({"ok": True, "pid": os.getpid()})
                break
            if fake:
                response = {"ok": True, "pid": os.getpid(), "command": command}
                if command == "replay":
                    response["us_per_call"] = float(request.get("value", 1.0))
                channel.send(response)
                continue
            if command == "prepare":
                result, state = _worker_init(
                    Path(request["source"]), Path(request["jit"]), request["host_report"],
                    request["case"], benchmark=request["benchmark"],
                )
                channel.send({"ok": True, "pid": os.getpid(), "result": result})
            elif command == "replay" and state is not None:
                channel.send({"ok": True, "pid": os.getpid(), "us_per_call": state[0]()})
            elif command == "inspect" and state is not None:
                channel.send({"ok": True, "pid": os.getpid(), "result": state[1]()})
            else:
                raise ValueError("invalid worker command or state")
        except Exception:
            channel.send({"ok": False, "pid": os.getpid(), "error": traceback.format_exc()})
            break


class Worker:
    def __init__(self, *, fake: bool, log: Path):
        parent, child = socket.socketpair()
        parent.settimeout(600)
        env = os.environ.copy()
        env["OPUS_CONTROL_FD"] = str(child.fileno())
        command = [sys.executable, "-m", "tasks.opus_a16w16_persistent.scored_candidate.dual_worker", "--worker"]
        if fake:
            command.append("--fake")
        descriptor = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.log = os.fdopen(descriptor, "w", encoding="utf-8")
        env["PYTHONPATH"] = str(REPO)
        self.process = subprocess.Popen(command, pass_fds=(child.fileno(),), env=env,
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        child.close()
        self.channel = Channel(parent)

    def call(self, command: str, **kwargs) -> dict:
        self.channel.send({"command": command, **kwargs})
        response = self.channel.recv()
        if response.get("pid") != self.process.pid or not response.get("ok"):
            raise RuntimeError(f"worker {self.process.pid} failed: {response.get('error', response)}")
        return response

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                try:
                    self.channel.sock.settimeout(10)
                    self.call("close")
                    self.process.wait(timeout=10)
                except (RuntimeError, EOFError, OSError, TimeoutError, subprocess.TimeoutExpired):
                    pass
        finally:
            self._signal_group(signal.SIGTERM)
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    self.process.wait(timeout=10)
            self.channel.sock.close()
            self.log.close()

    def _signal_group(self, signum: int) -> None:
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass


def paired_samples(baseline, candidate, repeats: int = REPEATS) -> dict:
    samples = {"aiter_us": [], "candidate_us": []}
    for index in range(repeats):
        order = (("aiter_us", baseline), ("candidate_us", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for key, worker in order:
            value = worker.call("replay")["us_per_call"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite GPU-event timing required")
            samples[key].append(value)
    return samples


def summarize(samples: dict) -> dict:
    import statistics

    if any(len(values) != REPEATS for values in samples.values()):
        raise ValueError("twenty paired samples required")
    base, cand = (samples[key] for key in ("aiter_us", "candidate_us"))
    base_med, cand_med = statistics.median(base), statistics.median(cand)
    if base_med <= 0 or cand_med <= 0:
        raise ValueError("positive medians required")
    return {"samples": samples, "aiter_median_us": base_med, "candidate_median_us": cand_med,
            "candidate_to_aiter_ratio": cand_med / base_med,
            "aiter_relative_mad": statistics.median(abs(v - base_med) for v in base) / base_med,
            "candidate_relative_mad": statistics.median(abs(v - cand_med) for v in cand) / cand_med}


def _gpu_pids() -> list[int]:
    from tasks.opus_a16w16_persistent.adversarial_admit import _gpu_pids as read_pids
    return read_pids()


def run_pair(args, task: dict, case: dict, host_report: str) -> dict:
    if _gpu_pids():
        raise RuntimeError("foreign active GPU process before workers")
    args.jit_baseline.mkdir(mode=0o700)
    args.jit_candidate.mkdir(mode=0o700)
    workers = []
    result = {"schema": "aiter-rs-opus-production-overlay-pair-v1", "case_id": case["id"],
              "kind": "unscored_production_overlay_control", "scored_eligible": False}
    try:
        for role in ("aiter", "candidate"):
            worker = Worker(fake=False, log=args.output.parent / f"{args.output.stem}-{role}.log")
            workers.append(worker)
        allowed = {worker.process.pid for worker in workers}
        result["worker_pids"] = {role: worker.process.pid for role, worker in zip(("aiter", "candidate"), workers)}
        for role, worker, source, jit in zip(
            ("aiter", "candidate"), workers,
            (args.aiter_source, args.overlay_tree), (args.jit_baseline, args.jit_candidate),
        ):
            response = worker.call("prepare", source=str(source), jit=str(jit),
                                   host_report=host_report, case=case, benchmark=not args.withheld)
            result[role] = response["result"]
            if (not result[role]["correctness_pass"] or
                (not args.withheld and not all((
                    result[role].get("pre_graph", {}).get("pass", False),
                    result[role].get("pre_timing", {}).get("pass", False),
                )))):
                result["pass"] = False
                result["status"] = "correctness_failed"
                return result
            active = _gpu_pids()
            if not active or not set(active).issubset(allowed):
                raise RuntimeError("foreign active GPU process during worker setup")
        result["same_boundary"] = True
        result["pid_gate"] = True
        if not args.withheld:
            if result["aiter"]["performance_input_hashes"] != result["candidate"]["performance_input_hashes"]:
                raise RuntimeError("production workers did not time identical BF16 inputs")
            samples = paired_samples(*workers)
            result["graph_performance"] = summarize(samples)
            result["post_graph"] = {}
            for role, worker in zip(("aiter", "candidate"), workers):
                result["post_graph"][role] = worker.call("inspect")["result"]
            active = _gpu_pids()
            result["pid_gate"] = bool(active) and set(active).issubset(allowed)
            result["graph_performance"]["noise_qualified"] = max(
                result["graph_performance"]["aiter_relative_mad"],
                result["graph_performance"]["candidate_relative_mad"],
            ) <= 0.05
            result["pass"] = all((result["post_graph"]["aiter"]["pass"],
                                  result["post_graph"]["candidate"]["pass"], result["pid_gate"],
                                  result["graph_performance"]["noise_qualified"],
                                  result["graph_performance"]["candidate_to_aiter_ratio"] <= 1.05))
            if not result["post_graph"]["aiter"]["pass"] or not result["post_graph"]["candidate"]["pass"]:
                result["status"] = "post_graph_correctness_failed"
            elif not result["pid_gate"] or not result["graph_performance"]["noise_qualified"]:
                result["status"] = "environment_or_noise_inconclusive"
            else:
                result["status"] = "single_bucket_pass" if result["pass"] else "single_bucket_performance_failed"
        else:
            result["pass"] = True
            result["status"] = "single_withheld_case_correctness_pass"
        return result
    finally:
        for worker in workers:
            worker.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fake", action="store_true")
    for name in ("aiter-source", "overlay-tree", "jit-baseline", "jit-candidate", "matrix", "host-gpu-report", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--candidate-header-sha256")
    parser.add_argument("--compiler-image-id")
    parser.add_argument("--withheld", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        fd = int(os.environ["OPUS_CONTROL_FD"])
        with socket.socket(fileno=fd) as sock:
            worker_loop(Channel(sock), fake=args.fake)
        return 0
    required = ("aiter_source", "overlay_tree", "jit_baseline", "jit_candidate", "matrix", "host_gpu_report", "output", "case", "candidate_header_sha256", "compiler_image_id")
    if any(getattr(args, name) is None for name in required):
        parser.error("all source, JIT, matrix, attestation, output, case, and header SHA arguments are required")
    for name in required:
        if name not in ("case", "candidate_header_sha256", "compiler_image_id"):
            setattr(args, name, getattr(args, name).resolve())
    task = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    public = json.loads((HERE / "public_matrix.json").read_text(encoding="utf-8"))
    validate_task(task, public)
    if args.compiler_image_id != task["compiler_image_id"]:
        raise ValueError("compiler image ID does not match trusted host attestation")
    if sha256(Path("/opt/rocm/bin/hipcc")) != task["compiler_hipcc_sha256"]:
        raise ValueError("hipcc binary hash changed")
    validate_runtime_paths(args.aiter_source, args.overlay_tree, args.jit_baseline, args.jit_candidate, args.output)
    sources = validate_sources(args.aiter_source, args.overlay_tree, args.candidate_header_sha256)
    matrix = read_matrix(args.matrix, task, withheld=args.withheld)
    case = select_case(matrix, args.case)
    if sha256(args.host_gpu_report) != task["host_gpu_report_sha256"]:
        raise ValueError("trusted host GPU attestation changed")
    try:
        result = run_pair(args, task, case, args.host_gpu_report.read_text(encoding="utf-8"))
    except Exception as error:
        result = {"schema": "aiter-rs-opus-production-overlay-pair-v1", "case_id": case["id"],
                  "kind": "unscored_production_overlay_control", "scored_eligible": False,
                  "pass": False, "status": "inconclusive_timeout_or_hang" if isinstance(error, TimeoutError)
                  else "inconclusive_execution_error", "error_type": type(error).__name__,
                  "error": str(error)}
    result.update({"aiter_sha": task["aiter_sha"], "source_check": sources,
                   "task_sha256": sha256(HERE / "task.json"),
                   "compiler_image_id": args.compiler_image_id,
                   "matrix_sha256": sha256(args.matrix),
                   "driver_sha256": sha256(Path(__file__)),
                   "host_gpu_report_sha256": sha256(args.host_gpu_report),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"case": case["id"] if not args.withheld else "withheld",
                      "pass": result["pass"], "raw_sha256": sha256(args.output)}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
