"""Trusted sequential eight-public then twelve-withheld OPUS control batch."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from .aggregate import aggregate_public, load_results
from .freeze import HERE, REPO, sha256, validate_matrix, validate_task

CASE_WATCHDOG_SECONDS = 1300
BATCH_BUDGET_SECONDS = 5400


def committed_public_cases() -> tuple[dict, list[dict]]:
    task = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    matrix = json.loads((HERE / "public_matrix.json").read_text(encoding="utf-8"))
    validate_task(task, matrix)
    return task, matrix["cases"]


def committed_withheld_cases(path: Path, task: dict) -> tuple[dict, list[dict]]:
    if path.resolve().is_relative_to(REPO) or sha256(path) != task["withheld_matrix_sha256"]:
        raise ValueError("withheld matrix location or commitment mismatch")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    validate_matrix(matrix, withheld=True)
    return matrix, matrix["cases"]


def candidate_sha(path: Path) -> str:
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 256 * 1024 or b"\x00" in path.read_bytes():
        raise ValueError("candidate must be a regular <=256KiB text header")
    return sha256(path)


def _private_json(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _run_case(script: Path, candidate: Path, case: dict, root: Path,
              env: dict, private_matrix: Path | None) -> int:
    command = ["bash", str(script), str(candidate), case["id"], str(root)]
    if private_matrix is not None:
        command.extend(["--withheld-matrix", str(private_matrix)])
    log = root.parent / (root.name + ".host.log")
    descriptor = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=CASE_WATCHDOG_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            return 124


def run_batch(args, *, run_case=_run_case) -> dict:
    task, public_cases = committed_public_cases()
    candidate = args.candidate_header.resolve()
    header_sha = candidate_sha(candidate)
    root = args.batch_root.resolve()
    study = args.study_root.resolve()
    code = args.code_root.resolve()
    if (root.exists() or not root.is_relative_to(study / "private")
            or root.is_relative_to(candidate.parent) or root.is_relative_to(code)):
        raise ValueError("batch root must be new, private, and outside source inputs")
    root.mkdir(mode=0o700)
    public_root = root / "public"
    public_root.mkdir(mode=0o700)
    env = os.environ.copy()
    env["AITERRS_STUDY_ROOT"] = str(study)
    env["AITERRS_CODE_ROOT"] = str(code)
    if args.host_gpu_report is not None:
        env["OPUS_HOST_GPU_REPORT"] = str(args.host_gpu_report.resolve())
    script = code / "tasks/opus_a16w16_persistent/scored_candidate/run_pair_host.sh"
    report = {"schema": "aiter-rs-opus-production-batch-host-v1",
              "task_sha256": sha256(HERE / "task.json"), "candidate_header_sha256": header_sha,
              "public_completed": 0, "withheld_completed": 0, "cases": [],
              "status": "running", "scored_eligible": False}
    source_root = root / "source"
    source_root.mkdir(mode=0o700)
    snapshot = source_root / "candidate-header.cuh"
    started = time.monotonic()

    def record_case(stage: str, case: dict, case_root: Path, exit_code: int, elapsed: float) -> None:
        entry = {"stage": stage, "case_id": case["id"], "exit_code": exit_code,
                 "wall_seconds": elapsed}
        for name in ("result.json", "attempt.json"):
            path = case_root / name
            if path.exists():
                entry[name.replace(".json", "_sha256")] = sha256(path)
        log = case_root.parent / (case_root.name + ".host.log")
        if log.exists():
            entry["host_log_sha256"] = sha256(log)
        report["cases"].append(entry)

    try:
        descriptor = os.open(snapshot, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(candidate.read_bytes())
        snapshot.chmod(0o400)
        if candidate_sha(snapshot) != header_sha or candidate_sha(candidate) != header_sha:
            raise RuntimeError("candidate changed while freezing private batch snapshot")
        for case in public_cases:
            if time.monotonic() - started > BATCH_BUDGET_SECONDS:
                report["status"] = "batch_budget_inconclusive"
                return report
            if candidate_sha(candidate) != header_sha or candidate_sha(snapshot) != header_sha:
                raise RuntimeError("candidate source changed during public batch")
            case_root = public_root / case["id"]
            case_started = time.monotonic()
            code_status = run_case(script, snapshot, case, case_root, env, None)
            record_case("public", case, case_root, code_status, time.monotonic() - case_started)
            if candidate_sha(candidate) != header_sha or candidate_sha(snapshot) != header_sha:
                raise RuntimeError("candidate source changed during public case")
            if code_status:
                report["status"] = "public_incomplete_or_failed"
                report["stopped_stage"] = "public"
                report["exit_code"] = code_status
                return report
            report["public_completed"] += 1
        gate = aggregate_public(load_results(public_root), task, header_sha)
        report["public_gate"] = gate
        if not gate["per_bucket_noninferior"]:
            report["status"] = "public_noninferiority_failed_or_inconclusive"
            return report

        # The withheld manifest is not read or mounted until every public case passes.
        matrix, hidden_cases = committed_withheld_cases(args.withheld_matrix.resolve(), task)
        withheld_root = root / "withheld"
        withheld_root.mkdir(mode=0o700)
        for case in hidden_cases:
            if time.monotonic() - started > BATCH_BUDGET_SECONDS:
                report["status"] = "batch_budget_inconclusive"
                return report
            if candidate_sha(candidate) != header_sha or candidate_sha(snapshot) != header_sha:
                raise RuntimeError("candidate source changed during withheld batch")
            if sha256(args.withheld_matrix) != task["withheld_matrix_sha256"]:
                raise RuntimeError("withheld matrix changed during batch")
            case_root = withheld_root / case["id"]
            case_started = time.monotonic()
            code_status = run_case(script, snapshot, case, case_root, env, args.withheld_matrix.resolve())
            record_case("withheld", case, case_root, code_status, time.monotonic() - case_started)
            if candidate_sha(candidate) != header_sha or candidate_sha(snapshot) != header_sha:
                raise RuntimeError("candidate source changed during withheld case")
            if code_status:
                report["status"] = "withheld_incomplete_or_failed"
                report["stopped_stage"] = "withheld"
                report["exit_code"] = code_status
                return report
            report["withheld_completed"] += 1
        from .aggregate import aggregate_withheld
        hidden_gate = aggregate_withheld(load_results(withheld_root), matrix, task, header_sha)
        report["withheld_gate"] = hidden_gate
        report["status"] = "full_matrix_ready_for_review"
        return report
    except Exception as error:
        report["status"] = "inconclusive_execution_error"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        return report
    finally:
        _private_json(root / "batch_report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-header", "study-root", "code-root", "batch-root", "withheld-matrix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--host-gpu-report", type=Path)
    args = parser.parse_args()
    report = run_batch(args)
    print(json.dumps({"status": report["status"], "public_completed": report["public_completed"],
                      "withheld_completed": report["withheld_completed"],
                      "scored_eligible": False, "private_batch_sha256": sha256(args.batch_root / "batch_report.json")}))
    return 0 if report["status"] == "full_matrix_ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
