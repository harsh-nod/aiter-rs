#!/usr/bin/env python3
"""Freeze and record isolated Codex HIP-kernel trials.

Raw trajectories stay local until reviewed for credentials and private data.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SUFFIXES = {".hip", ".cu", ".cuh", ".h", ".hpp", ".hh", ".cpp", ".cc", ".c", ".py", ".cmake"}
SOURCE_NAMES = {"CMakeLists.txt", "Makefile"}
SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".pytest_cache"}
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_FILES = 1024


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_task(path: Path) -> dict:
    task = json.loads(path.read_text(encoding="utf-8"))
    required = {"task_id", "task_revision", "aiter_sha", "gpu_sku", "target_arch", "operator_entry", "prompt_file", "starter_dir", "harness_revision", "task_mode"}
    missing = required - task.keys()
    if missing:
        raise ValueError(f"missing task fields: {', '.join(sorted(missing))}")
    if task["target_arch"] != "gfx950":
        raise ValueError("only gfx950 tasks are eligible")
    if task["task_mode"] not in {"no_feedback", "visible_tests_available"}:
        raise ValueError("task_mode must be no_feedback or visible_tests_available")
    if not all(isinstance(task[k], str) and task[k] for k in required):
        raise ValueError("required task fields must be nonempty strings")
    for key in ("task_id", "task_revision"):
        value = task[key]
        if value in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in value):
            raise ValueError(f"{key} must be a safe path component")
    if len(task["aiter_sha"]) != 40 or any(c not in "0123456789abcdef" for c in task["aiter_sha"].lower()):
        raise ValueError("aiter_sha must be a full 40-character Git SHA")
    for key in ("prompt_file", "starter_dir"):
        relative = Path(task[key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{key} must be relative to task directory")
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError(f"{key} escapes task directory")
        if not target.exists():
            raise ValueError(f"{key} does not exist: {target}")
    if not isinstance(task.get("build_sources", []), list) or not all(isinstance(source, str) for source in task.get("build_sources", [])):
        raise ValueError("build_sources must be a list of relative paths")
    for source in task.get("build_sources", []):
        relative = Path(source)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("build_sources must be relative paths within exported snapshots")
    if not isinstance(task.get("build_flags", []), list) or not all(isinstance(flag, str) for flag in task.get("build_flags", [])):
        raise ValueError("build_flags must be a list of argv values")
    for key in ("visible_checks", "hidden_checks"):
        checks = task.get(key, [])
        if not isinstance(checks, list) or any(not isinstance(c, list) or not c or not all(isinstance(x, str) for x in c) for c in checks):
            raise ValueError(f"{key} must be a list of argv arrays")
    if task["task_mode"] == "no_feedback" and task.get("visible_checks"):
        raise ValueError("no_feedback task cannot list visible checks")
    if task["task_mode"] == "visible_tests_available" and not task.get("visible_checks"):
        raise ValueError("visible_tests_available task requires visible checks")
    return task


def starter_hash(root: Path) -> str:
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"starter contains symlink: {path}")
        if path.is_file():
            items.append([str(path.relative_to(root)), digest(path.read_bytes())])
    return digest(canonical(items))


def freeze_payload(task_path: Path) -> dict:
    task = read_task(task_path)
    return {
        "schema": "aiter-rs-task-freeze-v1",
        "task_sha256": digest(task_path.read_bytes()),
        "prompt_sha256": digest((task_path.parent / task["prompt_file"]).read_bytes()),
        "starter_tree_sha256": starter_hash(task_path.parent / task["starter_dir"]),
        "task_id": task["task_id"],
        "task_revision": task["task_revision"],
        "harness_revision": task["harness_revision"],
        "aiter_sha": task["aiter_sha"],
        "gpu_sku": task["gpu_sku"],
        "target_arch": task["target_arch"],
        "task_mode": task["task_mode"],
        "harness_spec_sha256": task.get("harness_spec_sha256"),
        "build_sources": task.get("build_sources"),
        "build_flags": task.get("build_flags"),
    }


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.write("\n")


def agent_sandbox_command(cli: str, workspace: Path, model: str, reasoning_effort: str) -> tuple[list[str], dict[str, str], dict, list[Path]]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise ValueError("bwrap is required for agent execution; refusing an unsandboxed run")
    auth_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
    auth = auth_home / "auth.json"
    if not auth.is_file():
        raise ValueError("Codex auth.json is missing; refusing an unsandboxed run")
    ca_setting = os.environ.get("SSL_CERT_FILE") or os.environ.get("NODE_EXTRA_CA_CERTS")
    ca_file = Path(ca_setting).resolve() if ca_setting else None
    if ca_file is not None and not ca_file.is_file():
        raise ValueError("configured API CA certificate is missing")
    cli_binary = Path(cli).resolve()
    if not cli_binary.is_file():
        raise ValueError("agent CLI must resolve to a regular executable file")
    if not all(Path(path).is_dir() for path in ("/usr", "/etc")):
        raise ValueError("required read-only toolchain mounts are missing")
    rocm = Path("/opt/rocm").resolve()
    if not rocm.is_dir() or not rocm.is_relative_to(Path("/opt")) or rocm.parent != Path("/opt"):
        raise ValueError("ROCm must resolve to one toolchain directory under /opt")
    resolver = Path("/etc/resolv.conf").resolve()
    if not resolver.is_file():
        raise ValueError("system resolver file is missing; network API access cannot be qualified")
    command = [
        bwrap, "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--clearenv",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
        "--dir", "/opt", "--ro-bind", str(rocm), str(rocm),
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    ]
    if rocm != Path("/opt/rocm"):
        command.extend(("--symlink", rocm.name, "/opt/rocm"))
    if not resolver.is_relative_to(Path("/etc")) and not resolver.is_relative_to(Path("/usr")) and not resolver.is_relative_to(rocm):
        for parent in reversed(resolver.parent.parents):
            if parent != Path("/"):
                command.extend(("--dir", str(parent)))
        command.extend(("--dir", str(resolver.parent)))
        command.extend(("--ro-bind", str(resolver), str(resolver)))
    command.extend((
        "--dir", "/home", "--dir", "/home/agent", "--tmpfs", "/home/agent/.codex", "--dir", "/workspace",
        "--bind", str(workspace), "/workspace",
        "--ro-bind", str(cli_binary), "/codex",
    ))
    ephemeral_codex_files = [auth]
    for name in ("cloud-config-bundle-cache.json", "cloud-requirements-cache.json", "installation_id"):
        cache = auth_home / name
        if cache.is_file():
            ephemeral_codex_files.append(cache)
    for index, source in enumerate(ephemeral_codex_files):
        command.extend(("--perms", "0600", "--file", f"<fd:{index}>", f"/home/agent/.codex/{source.name}"))
    if ca_file is not None:
        command.extend(("--ro-bind", str(ca_file), "/home/agent/.codex/api-ca.pem"))
    command.extend((
        "--setenv", "HOME", "/home/agent",
        "--setenv", "CODEX_HOME", "/home/agent/.codex",
        "--setenv", "PATH", "/usr/bin:/bin:/opt/rocm/bin",
        "--setenv", "LANG", "C.UTF-8",
    ))
    if ca_file is not None:
        command.extend(("--setenv", "SSL_CERT_FILE", "/home/agent/.codex/api-ca.pem", "--setenv", "NODE_EXTRA_CA_CERTS", "/home/agent/.codex/api-ca.pem"))
    command.extend((
        "--chdir", "/workspace",
        "--", "/codex", "exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral", "--json",
        "-C", "/workspace", "-s", "workspace-write", "-c", "approval_policy=never",
        "-c", f"model_reasoning_effort={reasoning_effort}", "-m", model, "-",
    ))
    agent_env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    isolation = {
        "backend": "bwrap",
        "bwrap_version": subprocess.run([bwrap, "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip(),
        "mount_policy": "minimal-codex-hip-v1",
        "workspace": "read-write /workspace only",
        "system": "read-only /usr, /etc, selected ROCm toolchain; one read-only resolver file if needed",
        "auth": "Codex auth and policy cache copied by FD into per-run tmpfs CODEX_HOME; agent shell can read auth; no host writeback",
        "api_ca": "one read-only CA certificate, when configured",
        "home": "ephemeral /home/agent; host home and SSH keys not mounted",
        "tmp": "private tmpfs",
        "pid_ipc_uts": "unshared",
        "network": "shared for model API access; no SSH credentials or agent socket forwarded",
        "cli_sha256": file_digest(cli_binary),
    }
    return command, agent_env, isolation, ephemeral_codex_files


class Snapshotter:
    def __init__(self, workspace: Path, result_dir: Path):
        self.workspace = workspace
        self.result_dir = result_dir
        self.lock = threading.Lock()
        self.previous: dict[str, bytes] = {}
        self.previous_tree = None
        self.sequence = 0

    def capture(self, reason: str) -> None:
        with self.lock:
            files: dict[str, bytes] = {}
            for path in sorted(self.workspace.rglob("*")):
                if any(part in SKIP_DIRS for part in path.relative_to(self.workspace).parts):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                if path.suffix not in SOURCE_SUFFIXES and path.name not in SOURCE_NAMES:
                    continue
                if len(files) >= MAX_FILES:
                    raise RuntimeError("source-file limit exceeded")
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        raise RuntimeError(f"source-file limit exceeded: {path}")
                    files[str(path.relative_to(self.workspace))] = path.read_bytes()
                except FileNotFoundError:
                    continue
            hashes = {name: digest(data) for name, data in files.items()}
            tree = digest(canonical(hashes))
            if tree == self.previous_tree:
                return
            for name, data in files.items():
                blob = self.result_dir / "blobs" / hashes[name]
                if not blob.exists():
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with blob.open("xb") as out:
                            out.write(data)
                    except FileExistsError:
                        pass
            changes = []
            for name in sorted(set(self.previous) | set(files)):
                before, after = self.previous.get(name), files.get(name)
                if before == after:
                    continue
                changes.extend(difflib.unified_diff(
                    (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
                    (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
                    fromfile=f"a/{name}", tofile=f"b/{name}",
                ))
            self.sequence += 1
            diff_name = f"diffs/{self.sequence:06d}.patch"
            diff_path = self.result_dir / diff_name
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text("".join(changes), encoding="utf-8")
            entry = {
                "sequence": self.sequence,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "parent_tree_sha256": self.previous_tree,
                "tree_sha256": tree,
                "files": hashes,
                "diff": diff_name,
            }
            with (self.result_dir / "snapshots.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(entry, sort_keys=True) + "\n")
            self.previous, self.previous_tree = files, tree


def run_agent(args: argparse.Namespace) -> int:
    task_path = args.task.resolve()
    task = read_task(task_path)
    if args.unscored_preview and task.get("scored_eligible") is not False:
        raise ValueError("unscored preview requires scored_eligible=false")
    if not args.dry_run and not args.unscored_preview:
        if task.get("scored_eligible") is not True:
            raise ValueError("task must explicitly be marked scored_eligible to launch an agent")
        revision = task["harness_revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError("harness_revision must be a full Git SHA for agent launch")
        spec_sha = task.get("harness_spec_sha256", "")
        if len(spec_sha) != 64 or any(c not in "0123456789abcdef" for c in spec_sha.lower()):
            raise ValueError("harness_spec_sha256 must pin the scorer spec for agent launch")
        if not task.get("build_sources") or not task.get("build_flags"):
            raise ValueError("scored agent task needs frozen build_sources and build_flags")
    actual_freeze = freeze_payload(task_path)
    expected_freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if expected_freeze != actual_freeze:
        raise ValueError("task, prompt, or starter changed since freeze")
    if not args.replicate_id or args.replicate_id in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.replicate_id):
        raise ValueError("replicate_id must be a single safe path component")
    run_id = f"{task['task_id']}--{task['task_revision']}--{args.replicate_id}"
    result_dir = args.results.resolve() / run_id
    cli = shutil.which(args.cli)
    if cli is None:
        raise ValueError(f"agent CLI not found: {args.cli}")
    version = subprocess.run([cli, "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    command, agent_env, isolation, ephemeral_codex_files = agent_sandbox_command(cli, result_dir / "workspace", args.model, args.reasoning_effort)
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
    private_mounts = {str(codex_home / name): f"<host-codex-{name}>" for name in ("auth.json", "cloud-config-bundle-cache.json", "cloud-requirements-cache.json", "installation_id")}
    private_mounts[str(Path(cli).resolve())] = "<host-codex-binary>"
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("NODE_EXTRA_CA_CERTS"):
        private_mounts[str(Path(os.environ.get("SSL_CERT_FILE") or os.environ["NODE_EXTRA_CA_CERTS"]).resolve())] = "<host-api-ca-certificate>"
    private_mounts[str(result_dir / "workspace")] = "<host-agent-workspace>"
    safe_command = [private_mounts.get(item, item) for item in command]
    record = {
        "schema": "aiter-rs-agent-run-v1",
        "run_id": run_id,
        "task_freeze": actual_freeze,
        "task_freeze_sha256": digest(canonical(actual_freeze)),
        "agent": {"cli": args.cli, "version": version, "model": args.model, "reasoning_effort": args.reasoning_effort, "replicate_id": args.replicate_id, "sampling_seed_supported": False},
        "wall_seconds": args.wall_seconds,
        "command": safe_command,
        "isolation": isolation,
        "run_purpose": "command_preview" if args.dry_run else ("unscored_preview" if args.unscored_preview else "scored_trial_capture"),
        "incidence_eligible": not args.dry_run and not args.unscored_preview,
        "status": "dry_run" if args.dry_run else "running",
    }
    if args.dry_run:
        print(json.dumps(record, sort_keys=True, indent=2))
        return 0
    result_dir.mkdir(parents=True, exist_ok=False)
    workspace = result_dir / "workspace"
    shutil.copytree(task_path.parent / task["starter_dir"], workspace, symlinks=False)
    prompt = (task_path.parent / task["prompt_file"]).read_text(encoding="utf-8")
    feedback = ("No supplied correctness-test feedback is available during this attempt." if task["task_mode"] == "no_feedback" else "Visible checks are supplied in this workspace; you may run them during the attempt.")
    prompt = f"Implement the specified HIP kernel in this workspace. Target gfx950. {feedback} Do not attempt to access hidden tests.\n\n{prompt}"
    (result_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_json_exclusive(result_dir / "manifest.json", record)
    snapshotter = Snapshotter(workspace, result_dir)
    snapshotter.capture("starter")
    stop = threading.Event()
    snapshot_error: list[str] = []

    def poll_sources() -> None:
        while not stop.wait(0.25):
            try:
                snapshotter.capture("poll")
            except Exception as exc:
                snapshot_error.append(str(exc))
                return

    watcher = threading.Thread(target=poll_sources, daemon=True)
    watcher.start()
    started = time.monotonic()
    timed_out = False
    with (result_dir / "stderr.log").open("wb") as err, (result_dir / "events.jsonl").open("wb") as events:
        # bwrap copies these FDs into its private tmpfs, so token refresh never writes to the host profile.
        file_descriptors = []
        try:
            for path in ephemeral_codex_files:
                file_descriptors.append(os.open(path, os.O_RDONLY))
            executable_command = [str(file_descriptors[int(item[4:-1])]) if item.startswith("<fd:") and item.endswith(">") else item for item in command]
            process = subprocess.Popen(executable_command, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, start_new_session=True, env=agent_env, pass_fds=tuple(file_descriptors))
        finally:
            for descriptor in file_descriptors:
                os.close(descriptor)
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(prompt.encode("utf-8"))
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        pending = b""
        while True:
            remaining = args.wall_seconds - (time.monotonic() - started)
            if remaining <= 0 and process.poll() is None:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            ready = selector.select(timeout=min(0.25, max(0.0, remaining)))
            for key, _ in ready:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    events.write(line + b"\n")
                    events.flush()
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and event.get("type", "").endswith("completed"):
                            snapshotter.capture("agent_event")
                    except (json.JSONDecodeError, RuntimeError) as exc:
                        snapshot_error.append(str(exc))
            if process.poll() is not None and not selector.get_map():
                break
        if pending:
            events.write(pending)
        selector.close()
        process.stdout.close()
    stop.set()
    watcher.join(timeout=2)
    try:
        snapshotter.capture("final")
    except Exception as exc:
        snapshot_error.append(str(exc))
    final = {
        "status": "timeout" if timed_out else ("completed" if process.returncode == 0 else "agent_failed"),
        "agent_exit_code": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "snapshot_count": snapshotter.sequence,
        "final_tree_sha256": snapshotter.previous_tree,
        "snapshot_errors": snapshot_error,
        "scored": False,
        "incidence_eligible": record["incidence_eligible"],
        "scoring_note": "No correctness or performance result is implied by an agent CLI exit code.",
    }
    write_json_exclusive(result_dir / "result.json", final)
    print(json.dumps({"run_dir": str(result_dir), **final}, sort_keys=True))
    return 0 if final["status"] == "completed" and not snapshot_error else 1


def checked_git_head(root: Path, expected: str, scopes: list[str]) -> None:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    if head != expected:
        raise ValueError(f"checkout at {root} is {head}, expected {expected}")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", *scopes], capture_output=True, text=True, check=True, timeout=15)
    if status.stdout.strip():
        raise ValueError(f"checkout is not clean in {scopes}: {root}")


def export_snapshot(run_dir: Path, snapshot: dict, destination: Path) -> None:
    files = snapshot["files"]
    if digest(canonical(files)) != snapshot["tree_sha256"]:
        raise ValueError("snapshot tree hash does not match file manifest")
    destination.mkdir(parents=True, exist_ok=False)
    for name, sha in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError(f"unsafe snapshot path: {name}")
        blob = run_dir / "blobs" / sha
        data = blob.read_bytes()
        if digest(data) != sha:
            raise ValueError(f"snapshot blob hash mismatch: {name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(data)


def run_limited(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, wall_seconds: int, env: dict[str, str] | None = None) -> dict:
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True, env=env)
        try:
            exit_code = process.wait(timeout=wall_seconds)
            status = "completed" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code, status = process.returncode, "timeout"
    return {"status": status, "exit_code": exit_code, "elapsed_seconds": round(time.monotonic() - started, 3)}


def validate_private_inputs(spec_path: Path, withheld_path: Path, host_report_path: Path, task: dict, workspace: Path, harness_root: Path) -> tuple[str, str]:
    if withheld_path.is_relative_to(workspace.parent) or host_report_path.is_relative_to(workspace.parent):
        raise ValueError("private scoring inputs must stay outside the agent run directory")
    if withheld_path.is_relative_to(harness_root):
        raise ValueError("withheld manifest must stay outside the public harness checkout")
    public = json.loads(spec_path.read_text(encoding="utf-8"))
    if any(public.get(key) != task[key] for key in ("task_id", "aiter_sha", "gpu_sku", "target_arch")):
        raise ValueError("public scorer spec does not match frozen task identity")
    commitment = public.get("withheld_cases_sha256")
    if not isinstance(commitment, str) or len(commitment) != 64:
        raise ValueError("public scorer spec lacks a withheld-case SHA256 commitment")
    withheld_raw = withheld_path.read_bytes()
    if digest(withheld_raw) != commitment:
        raise ValueError("withheld-case SHA256 does not match public spec")
    withheld = json.loads(withheld_raw)
    if withheld.get("schema_version") != 1 or withheld.get("task_id") != task["task_id"]:
        raise ValueError("withheld manifest has wrong schema or task ID")
    cases = withheld.get("cases", [])
    if not cases or any(case.get("visibility") != "withheld" for case in cases):
        raise ValueError("private manifest needs withheld cases")
    report_raw = host_report_path.read_bytes()
    report = json.loads(report_raw)
    if report.get("schema_version") != 1 or report.get("gpu_name") != task["gpu_sku"] or report.get("arch") != task["target_arch"] or report.get("card_model") != public.get("gpu_pci_device_id"):
        raise ValueError("host GPU report does not match frozen gfx950 task")
    return commitment, digest(report_raw)


def validate_harness_result(path: Path, task: dict, spec_sha: str, binary_sha: str, freeze_sha: str, tree_sha: str, withheld_sha: str, host_report_sha: str) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"scorer result missing or invalid: {exc}") from exc
    expected = {
        "task_id": task["task_id"],
        "aiter_sha": task["aiter_sha"],
        "spec_sha256": spec_sha,
        "candidate_path_sha256": binary_sha,
        "task_freeze_sha256": freeze_sha,
        "final_tree_sha256": tree_sha,
        "withheld_cases_sha256": withheld_sha,
        "withheld_cases_evaluated": True,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"scorer result {field} mismatch")
    if not isinstance(result.get("joint_pass"), bool) or not isinstance(result.get("environment"), dict):
        raise ValueError("scorer result lacks joint_pass or environment")
    if result["environment"].get("host_gpu_report_sha256") != host_report_sha:
        raise ValueError("scorer result host_gpu_report_sha256 mismatch")
    return result


def score_snapshots(args: argparse.Namespace) -> int:
    task_path = args.task.resolve()
    task = read_task(task_path)
    freeze = freeze_payload(task_path)
    if json.loads(args.freeze.read_text(encoding="utf-8")) != freeze:
        raise ValueError("task changed since freeze")
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("incidence_eligible") is not True:
        raise ValueError("unscored previews cannot enter the scored harness pipeline")
    if manifest["task_freeze"] != freeze:
        raise ValueError("run manifest refers to a different frozen task")
    if not (run_dir / "result.json").is_file():
        raise ValueError("agent run has not ended; result.json is missing")
    harness_root = args.harness_root.resolve()
    spec = args.spec.resolve()
    aiter_source = args.aiter_source.resolve()
    withheld_spec = args.withheld_spec.resolve()
    host_gpu_report = args.host_gpu_report.resolve()
    output = args.output.resolve()
    workspace = (run_dir / "workspace").resolve()
    if output.is_relative_to(workspace) or harness_root.is_relative_to(workspace) or aiter_source.is_relative_to(workspace):
        raise ValueError("trusted scorer paths must remain outside the agent workspace")
    if not spec.is_relative_to(harness_root):
        raise ValueError("scorer spec must be in the pinned harness checkout")
    expected_spec = task.get("harness_spec_sha256")
    if not expected_spec or digest(spec.read_bytes()) != expected_spec:
        raise ValueError("scorer spec differs from frozen task")
    withheld_sha, host_report_sha = validate_private_inputs(spec, withheld_spec, host_gpu_report, task, workspace, harness_root)
    checked_git_head(harness_root, task["harness_revision"], ["harness", "references"])
    checked_git_head(aiter_source, task["aiter_sha"], ["."])
    tracked = subprocess.run(["git", "-C", str(harness_root), "ls-files", "--error-unmatch", str(spec.relative_to(harness_root))], capture_output=True, timeout=15)
    if tracked.returncode != 0:
        raise ValueError("scorer spec must be tracked in pinned harness revision")
    snapshots = [json.loads(line) for line in (run_dir / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    if not snapshots:
        raise ValueError("run contains no source snapshots")
    if not task.get("build_sources") or not task.get("build_flags"):
        raise ValueError("scoring requires frozen build_sources and build_flags")
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise ValueError(f"HIP compiler not found: {args.compiler}")
    compiler_version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    if args.snapshot == "final":
        selected = [snapshots[-1]]
    elif args.snapshot == "all":
        selected = snapshots
    else:
        selected = [next((item for item in snapshots if item["sequence"] == int(args.snapshot)), None)]
        if selected[0] is None:
            raise ValueError(f"snapshot {args.snapshot} does not exist")
    score_manifest = {
        "schema": "aiter-rs-post-agent-score-v1",
        "run_id": manifest["run_id"],
        "task_freeze_sha256": manifest["task_freeze_sha256"],
        "harness_revision": task["harness_revision"],
        "harness_spec_sha256": expected_spec,
        "withheld_cases_sha256": withheld_sha,
        "host_gpu_report_sha256": host_report_sha,
        "aiter_sha": task["aiter_sha"],
        "gpu_sku": task["gpu_sku"],
        "target_arch": task["target_arch"],
        "task_mode": task["task_mode"],
        "compiler": compiler,
        "compiler_version": compiler_version,
        "build_sources": task["build_sources"],
        "build_flags": task["build_flags"],
        "snapshot_selection": args.snapshot,
        "correctness_only": args.correctness_only,
        "results": [],
    }
    if args.dry_run:
        print(json.dumps({**score_manifest, "selected_snapshots": [item["sequence"] for item in selected], "status": "dry_run"}, sort_keys=True, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for snapshot in selected:
        sequence = snapshot["sequence"]
        item_dir = output / f"snapshot-{sequence:06d}"
        item_dir.mkdir()
        candidate = item_dir / "candidate"
        export_snapshot(run_dir, snapshot, candidate)
        item = {"sequence": sequence, "tree_sha256": snapshot["tree_sha256"]}
        source_paths = [candidate / source for source in task["build_sources"]]
        if not all(path.is_file() for path in source_paths):
            item["status"] = "compile_input_missing"
            write_json_exclusive(item_dir / "score_record.json", item)
            score_manifest["results"].append(item)
            continue
        library = item_dir / "libcandidate.so"
        build_command = [compiler, *task["build_flags"], *(str(path) for path in source_paths), "-o", str(library)]
        build = run_limited(build_command, item_dir, item_dir / "compile.stdout.log", item_dir / "compile.stderr.log", args.build_wall_seconds)
        item["build_command"] = build_command
        item["build"] = build
        if build["status"] != "completed" or not library.is_file():
            item["status"] = "compile_timeout" if build["status"] == "timeout" else "compile_failed"
            write_json_exclusive(item_dir / "score_record.json", item)
            score_manifest["results"].append(item)
            continue
        item["binary_sha256"] = digest(library.read_bytes())
        scorer_output = item_dir / "harness"
        command = [sys.executable, "-m", "harness.run", "--spec", str(spec), "--candidate", str(library), "--aiter-source", str(aiter_source), "--output", str(scorer_output), "--withheld-spec", str(withheld_spec), "--host-gpu-report", str(host_gpu_report), "--task-freeze-sha256", manifest["task_freeze_sha256"], "--final-tree-sha256", snapshot["tree_sha256"]]
        if args.correctness_only:
            command.append("--correctness-only")
        scorer_env = os.environ.copy()
        scorer_env["PYTHONDONTWRITEBYTECODE"] = "1"
        scored = run_limited(command, harness_root, item_dir / "scorer.stdout.log", item_dir / "scorer.stderr.log", args.score_wall_seconds, scorer_env)
        item["status"] = "scorer_timeout" if scored["status"] == "timeout" else "scorer_result_invalid"
        item["scorer"] = scored
        item["scorer_command"] = ["<private-withheld-spec>" if arg == str(withheld_spec) else ("<private-host-gpu-report>" if arg == str(host_gpu_report) else arg) for arg in command]
        if scored["status"] != "timeout":
            try:
                result = validate_harness_result(scorer_output / "result.json", task, expected_spec, item["binary_sha256"], manifest["task_freeze_sha256"], snapshot["tree_sha256"], withheld_sha, host_report_sha)
                item["status"] = "completed"
                item["joint_pass"] = result["joint_pass"]
            except ValueError as exc:
                item["validation_error"] = str(exc)
        write_json_exclusive(item_dir / "score_record.json", item)
        score_manifest["results"].append(item)
    write_json_exclusive(output / "score_manifest.json", score_manifest)
    print(json.dumps({"output": str(output), "snapshot_count": len(selected), "statuses": [item["status"] for item in score_manifest["results"]]}, sort_keys=True))
    return 0 if all(item["status"] == "completed" for item in score_manifest["results"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze", help="write an immutable task input manifest")
    freeze.add_argument("task", type=Path)
    freeze.add_argument("output", type=Path)
    run = sub.add_parser("run", help="invoke one agent session and capture raw trajectory")
    run.add_argument("task", type=Path)
    run.add_argument("--freeze", type=Path, required=True)
    run.add_argument("--results", type=Path, default=Path(__file__).parent / "artifacts")
    run.add_argument("--replicate-id", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    run.add_argument("--cli", default="codex")
    run.add_argument("--wall-seconds", type=int, default=1800)
    preview = run.add_mutually_exclusive_group()
    preview.add_argument("--dry-run", action="store_true")
    preview.add_argument("--unscored-preview", action="store_true", help="run a non-eligible exploratory task without incidence scoring")
    score = sub.add_parser("score", help="score immutable snapshots after the agent has ended")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--task", type=Path, required=True)
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--harness-root", type=Path, required=True)
    score.add_argument("--spec", type=Path, required=True)
    score.add_argument("--aiter-source", type=Path, required=True)
    score.add_argument("--withheld-spec", type=Path, required=True)
    score.add_argument("--host-gpu-report", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--snapshot", default="all", help="all, final, or one sequence number")
    score.add_argument("--score-wall-seconds", type=int, default=300)
    score.add_argument("--build-wall-seconds", type=int, default=180)
    score.add_argument("--compiler", default="hipcc")
    score.add_argument("--correctness-only", action="store_true")
    score.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        write_json_exclusive(args.output, freeze_payload(args.task.resolve()))
        print(args.output)
        return 0
    if args.action == "run":
        if args.wall_seconds < 1:
            raise ValueError("wall-seconds must be positive")
        return run_agent(args)
    if args.score_wall_seconds < 1 or args.build_wall_seconds < 1:
        raise ValueError("score and build wall times must be positive")
    return score_snapshots(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"runner: {exc}", file=sys.stderr)
        sys.exit(2)
