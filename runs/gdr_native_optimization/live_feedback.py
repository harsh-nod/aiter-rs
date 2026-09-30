"""Trusted live public feedback for the GDR optimization task.

The agent sees only the workspace request/response files. SSH credentials,
Docker, raw scorer output, and the remote private directory stay on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Callable

from runs.gdr_native_optimization.boundary import (
    LIVE_TASK_DIR,
    SCORED_TASK_DIR,
    RESPONSE_SCHEMA,
    process_request,
    public_container_command,
    sha256,
    starter_hash,
    validate_source_tree,
    validate_task,
)


def _write_private(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)


class LiveFeedbackBroker:
    def __init__(self, workspace: Path, private_root: Path, task_dir: Path,
                 score_public: Callable[[Path, str], dict], capture_source: Callable[[str], None] | None = None,
                 poll_seconds: float = 0.05):
        task, contract = validate_task(task_dir)
        if task["task_mode"] != "brokered_public_feedback":
            raise ValueError("live feedback requires the brokered task revision")
        self.workspace = workspace.resolve()
        self.private_root = private_root.resolve()
        if self.private_root.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.private_root):
            raise ValueError("private broker results must be disjoint from the agent workspace")
        if self.private_root.is_relative_to(task_dir.resolve().parents[2]):
            raise ValueError("private broker results must stay outside the repository")
        self.private_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.requests = self.workspace / ".feedback/requests"
        self.responses = self.workspace / ".feedback/responses"
        self.requests.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.responses.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.task_dir = task_dir
        self.score_public = score_public
        self.capture_source = capture_source
        self.poll_seconds = poll_seconds
        self.max_requests = contract["max_feedback_requests"]
        self.processed = 0
        self.errors: list[str] = []
        self.unprocessed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gdr-public-feedback", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> dict:
        self._stop.set()
        self._thread.join()
        known = {f"{number:04d}.json" for number in range(1, self.processed + 1)}
        self.unprocessed = sum(path.name not in known for path in self.requests.glob("*.json"))
        events = self.private_root / "feedback_events.jsonl"
        status = ("candidate_timeout_or_environment_ambiguous"
                  if any(name in {"PublicScoreTimeout", "TimeoutExpired"} for name in self.errors)
                  else "complete" if not self.errors and not self.unprocessed else "protocol_error")
        return {
            "schema": "aiter-rs-gdr-live-feedback-summary-v1",
            "request_count": self.processed,
            "request_limit": self.max_requests,
            "broker_error_classes": self.errors,
            "unprocessed_request_count": self.unprocessed,
            "private_event_log_sha256": sha256(events) if events.is_file() else None,
            "status": status,
        }

    def _run(self) -> None:
        for request_id in range(1, self.max_requests + 1):
            request = self.requests / f"{request_id:04d}.json"
            while not request.exists():
                if self._stop.wait(self.poll_seconds):
                    return
            self.processed += 1
            try:
                if self.capture_source is not None:
                    self.capture_source(f"feedback_request_{request_id:04d}")
                response = process_request(self.workspace, self.private_root, request_id,
                                           self.score_public, self.task_dir)
                if response["status"] == "error":
                    raw = json.loads((self.private_root / f"raw-{request_id:04d}.json").read_text())
                    self.errors.append(raw.get("broker_error_class", "PublicScoreUnresolved"))
                elif response["status"] == "candidate_timeout_or_environment_ambiguous":
                    self.errors.append("PublicScoreTimeout")
            except Exception as exc:
                self.errors.append(type(exc).__name__)
                self._reject(request, request_id, exc)

    def _reject(self, request: Path, request_id: int, exc: Exception) -> None:
        response = {
            "schema": RESPONSE_SCHEMA,
            "request_id": request_id,
            "source_sha256": None,
            "status": "error",
            "visible_case_results": {},
            "benchmark_bucket_results": {},
            "raw_result_sha256": None,
        }
        target = self.responses / f"{request_id:04d}.json"
        try:
            if self.responses.is_symlink() or target.is_symlink():
                raise ValueError("unsafe feedback response path")
            _write_private(target, json.dumps(response, sort_keys=True) + "\n")
            response_sha = sha256(target)
        except Exception as response_exc:
            self.errors.append(type(response_exc).__name__)
            response_sha = None
        event = {
            "request_id": request_id,
            "request_sha256": sha256(request) if request.is_file() and not request.is_symlink() else None,
            "response_sha256": response_sha,
            "error_class": type(exc).__name__,
        }
        with (self.private_root / "feedback_events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, sort_keys=True) + "\n")


_SAFE_REMOTE = re.compile(r"/[A-Za-z0-9._/-]+\Z")
_SAFE_HOST = re.compile(r"[A-Za-z0-9._-]+\Z")
_RESULT_MARKER = re.compile(r"^AITERRS_PUBLIC_RESULT_SHA256=([0-9a-f]{64})$", re.MULTILINE)


class PublicScoreTimeout(RuntimeError):
    pass


class RemotePublicScorer:
    def __init__(self, config_path: Path, private_root: Path, run_id: str,
                 task_dir: Path = LIVE_TASK_DIR):
        config_path = config_path.resolve()
        private_root = private_root.resolve()
        if not config_path.is_file() or config_path.is_relative_to(private_root):
            raise ValueError("trusted remote feedback config is missing or inside agent results")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {"schema", "ssh_host", "remote_root", "remote_repo", "remote_aiter",
                    "remote_host_gpu_report", "remote_repo_head", "image", "public_wall_seconds"}
        if set(config) != expected or config["schema"] != "aiter-rs-gdr-public-ssh-v1":
            raise ValueError("remote public scorer config has unexpected fields")
        if not _SAFE_HOST.fullmatch(config["ssh_host"]):
            raise ValueError("unsafe SSH host alias")
        for name in ("remote_root", "remote_repo", "remote_aiter", "remote_host_gpu_report"):
            if not isinstance(config[name], str) or not _SAFE_REMOTE.fullmatch(config[name]) or ".." in Path(config[name]).parts:
                raise ValueError(f"unsafe remote path: {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", config["remote_repo_head"]):
            raise ValueError("remote repository HEAD must be pinned")
        if not isinstance(config["public_wall_seconds"], int) or not 30 <= config["public_wall_seconds"] <= 600:
            raise ValueError("public scorer wall limit must be 30-600 seconds")
        if not isinstance(config["image"], str) or not config["image"]:
            raise ValueError("public scorer image is missing")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("unsafe run ID")
        task_dir = task_dir.resolve()
        if task_dir not in {LIVE_TASK_DIR, SCORED_TASK_DIR}:
            raise ValueError("trusted public broker task must be a frozen live revision")
        validate_task(task_dir)
        self.config = config
        self.task_dir = task_dir
        self.config_sha256 = sha256(config_path)
        self.private_root = private_root
        self.remote_run = f"{config['remote_root'].rstrip('/')}/{run_id}"
        self._remote(["mkdir", "-m", "700", self.remote_run], timeout=30)

    def _remote(self, argv: list[str], timeout: int, stdout_path: Path | None = None,
                stderr_path: Path | None = None) -> str:
        command = ["ssh", "-T", "-o", "BatchMode=yes", self.config["ssh_host"], shlex.join(argv)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if stdout_path is not None:
            _write_private(stdout_path, result.stdout)
        if stderr_path is not None:
            _write_private(stderr_path, result.stderr)
        if result.returncode != 0:
            if "AITERRS_PUBLIC_TIMEOUT" in result.stdout:
                raise PublicScoreTimeout("trusted public scorer watchdog timed out")
            raise RuntimeError(f"trusted remote public scorer command failed with exit {result.returncode}")
        return result.stdout

    def __call__(self, snapshot: Path, kind: str) -> dict:
        if kind not in {"correctness", "benchmark"} or not re.fullmatch(r"[0-9]{4}", snapshot.name):
            raise ValueError("invalid public feedback snapshot or kind")
        number = snapshot.name
        remote_request = f"{self.remote_run}/{number}"
        self._remote(["mkdir", "-m", "700", remote_request], timeout=30)
        upload = subprocess.run(
            ["scp", "-q", "-r", str(snapshot),
             f"{self.config['ssh_host']}:{remote_request}/snapshot"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        _write_private(self.private_root / f"upload-{number}.stderr.log", upload.stderr)
        if upload.returncode != 0:
            raise RuntimeError("trusted public snapshot transfer failed")
        expected_tree = starter_hash(snapshot)
        remote_command = [
            "python3", "-m", "runs.gdr_native_optimization.live_feedback", "remote-score",
            "--snapshot", f"{remote_request}/snapshot",
            "--expected-snapshot-sha256", expected_tree,
            "--repo", self.config["remote_repo"],
            "--expected-repo-head", self.config["remote_repo_head"],
            "--aiter-source", self.config["remote_aiter"],
            "--host-gpu-report", self.config["remote_host_gpu_report"],
            "--task-dir", f"{self.config['remote_repo']}/runs/tasks/{self.task_dir.name}",
            "--request-root", remote_request,
            "--image", self.config["image"],
            "--kind", kind,
            "--wall-seconds", str(self.config["public_wall_seconds"]),
        ]
        output = self._remote(
            ["env", f"PYTHONPATH={self.config['remote_repo']}", *remote_command],
            timeout=self.config["public_wall_seconds"] + 120,
            stdout_path=self.private_root / f"remote-{number}.stdout.log",
            stderr_path=self.private_root / f"remote-{number}.stderr.log",
        )
        marker = _RESULT_MARKER.search(output)
        if marker is None:
            raise RuntimeError("trusted remote scorer omitted its result commitment")
        local_raw = self.private_root / f"remote-{number}.json"
        download = subprocess.run(
            ["scp", "-q", f"{self.config['ssh_host']}:{remote_request}/public/result.json", str(local_raw)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        _write_private(self.private_root / f"download-{number}.stderr.log", download.stderr)
        if download.returncode != 0 or not local_raw.is_file() or sha256(local_raw) != marker.group(1):
            raise RuntimeError("trusted remote result transfer or SHA256 verification failed")
        if starter_hash(snapshot) != expected_tree:
            raise RuntimeError("public source changed after trusted remote scoring")
        raw = json.loads(local_raw.read_text(encoding="utf-8"))
        task, _ = validate_task(self.task_dir)
        expected = {
            "schema": "aiter-rs-gdr-opt-public-raw-v1",
            "source_sha256": sha256(snapshot / "kernel.hip"),
            "abi_header_sha256": task["abi_header_sha256"],
            "compiler_hipcc_sha256": task["compiler_hipcc_sha256"],
            "aiter_sha": task["aiter_sha"],
            "public_spec_sha256": task["harness_spec_sha256"],
            "large_fixture_sha256": task["large_fixture_sha256"],
        }
        if raw.get("schema") != expected["schema"]:
            raise ValueError("remote public result schema differs from pin")
        for name, value in expected.items():
            if name == "schema":
                continue
            required = raw.get("status") not in {"setup_error", "environment_invalid"}
            if (required or name in raw) and raw.get(name) != value:
                raise ValueError(f"remote public result {name} differs from pin")
        if raw.get("status") in {"complete", "correctness_failed"}:
            if (raw.get("host_gpu_report_sha256") != task["host_gpu_report_sha256"] or
                    raw.get("arch") != task["target_arch"] or
                    "0x75a0" not in raw.get("rocm_product", "")):
                raise ValueError("remote public GPU attestation differs from task pin")
        return raw


def remote_score(args: argparse.Namespace) -> None:
    from runs.gdr_native_optimization.post_agent import _run_owned_container

    snapshot, repo, aiter, report, task_dir, request_root = (
        path.resolve() for path in (args.snapshot, args.repo, args.aiter_source,
                                    args.host_gpu_report, args.task_dir, args.request_root)
    )
    if task_dir not in {LIVE_TASK_DIR, SCORED_TASK_DIR} or task_dir != repo / "runs/tasks" / task_dir.name:
        raise ValueError("remote task must be the pinned live revision in the trusted checkout")
    if request_root.is_relative_to(repo) or request_root.is_relative_to(aiter) or not snapshot.is_relative_to(request_root):
        raise ValueError("remote request must be private and separate from trusted source")
    if report.is_relative_to(request_root) or report.is_relative_to(repo):
        raise ValueError("host GPU attestation must be outside source and request mounts")
    task, _ = validate_task(task_dir)
    if task["task_mode"] != "brokered_public_feedback" or (
        task["scored_eligible"] is not (task_dir == SCORED_TASK_DIR)
    ):
        raise ValueError("remote public scorer requires a frozen live broker task")
    if subprocess.check_output(["git", "-c", f"safe.directory={repo}", "-C", str(repo),
                                "rev-parse", "HEAD"], text=True, timeout=30).strip() != args.expected_repo_head:
        raise ValueError("remote trusted checkout differs from the session pin")
    if subprocess.check_output(["git", "-c", f"safe.directory={aiter}", "-C", str(aiter),
                                "rev-parse", "HEAD"], text=True, timeout=30).strip() != task["aiter_sha"]:
        raise ValueError("remote AITER checkout differs from the task pin")
    if subprocess.check_output(["git", "-c", f"safe.directory={aiter}", "-C", str(aiter),
                                "status", "--porcelain", "--untracked-files=all"], text=True, timeout=30).strip():
        raise ValueError("remote AITER checkout is dirty")
    if starter_hash(snapshot) != args.expected_snapshot_sha256:
        raise ValueError("remote public snapshot SHA256 differs from uploaded source")
    fixed = {
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    }
    validate_source_tree(snapshot, fixed)
    if sha256(report) != task["host_gpu_report_sha256"]:
        raise ValueError("remote host GPU report differs from task pin")
    output = request_root / "public"
    output.mkdir(mode=0o700, exist_ok=False)
    command = public_container_command(snapshot, repo, aiter, output, report,
                                       task, args.image, args.kind, task_dir)
    run = _run_owned_container(command, "public", request_root, repo, args.wall_seconds)
    if starter_hash(snapshot) != args.expected_snapshot_sha256:
        raise ValueError("remote public snapshot changed during scoring")
    result = output / "result.json"
    if run["status"] == "timeout":
        print("AITERRS_PUBLIC_TIMEOUT", flush=True)
        raise PublicScoreTimeout("remote public scorer watchdog timed out")
    if not result.is_file():
        raise RuntimeError("remote public scorer did not write a bounded result")
    print(f"AITERRS_PUBLIC_RESULT_SHA256={sha256(result)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    remote = sub.add_parser("remote-score")
    for name in ("snapshot", "repo", "aiter-source", "host-gpu-report", "task-dir", "request-root"):
        remote.add_argument(f"--{name}", required=True, type=Path)
    remote.add_argument("--expected-snapshot-sha256", required=True)
    remote.add_argument("--expected-repo-head", required=True)
    remote.add_argument("--image", required=True)
    remote.add_argument("--kind", choices=("correctness", "benchmark"), required=True)
    remote.add_argument("--wall-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.command == "remote-score":
        remote_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
