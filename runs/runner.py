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


def read_task(path: Path) -> dict:
    task = json.loads(path.read_text(encoding="utf-8"))
    required = {"task_id", "task_revision", "aiter_sha", "gpu_sku", "target_arch", "operator_entry", "prompt_file", "starter_dir", "harness_revision"}
    missing = required - task.keys()
    if missing:
        raise ValueError(f"missing task fields: {', '.join(sorted(missing))}")
    if task["target_arch"] != "gfx950":
        raise ValueError("only gfx950 tasks are eligible")
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
    for key in ("visible_checks", "hidden_checks"):
        checks = task.get(key, [])
        if not isinstance(checks, list) or any(not isinstance(c, list) or not c or not all(isinstance(x, str) for x in c) for c in checks):
            raise ValueError(f"{key} must be a list of argv arrays")
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
    }


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.write("\n")


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
    if not args.dry_run:
        if task.get("scored_eligible") is not True:
            raise ValueError("task must explicitly be marked scored_eligible to launch an agent")
        revision = task["harness_revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError("harness_revision must be a full Git SHA for agent launch")
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
    command = [cli, "exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral", "--json", "-C", str(result_dir / "workspace"), "-s", "workspace-write", "-c", "approval_policy=never", "-c", f"model_reasoning_effort={args.reasoning_effort}", "-m", args.model, "-"]
    record = {
        "schema": "aiter-rs-agent-run-v1",
        "run_id": run_id,
        "task_freeze": actual_freeze,
        "task_freeze_sha256": digest(canonical(actual_freeze)),
        "agent": {"cli": args.cli, "version": version, "model": args.model, "reasoning_effort": args.reasoning_effort, "replicate_id": args.replicate_id, "sampling_seed_supported": False},
        "wall_seconds": args.wall_seconds,
        "command": command,
        "status": "dry_run" if args.dry_run else "running",
    }
    if args.dry_run:
        print(json.dumps(record, sort_keys=True, indent=2))
        return 0
    result_dir.mkdir(parents=True, exist_ok=False)
    workspace = result_dir / "workspace"
    shutil.copytree(task_path.parent / task["starter_dir"], workspace, symlinks=False)
    prompt = (task_path.parent / task["prompt_file"]).read_text(encoding="utf-8")
    prompt = f"Implement the specified HIP kernel in this workspace. Target gfx950. You may use the visible tests provided in the workspace. Do not attempt to access hidden tests.\n\n{prompt}"
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
        process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, start_new_session=True)
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
        "scoring_note": "No correctness or performance result is implied by an agent CLI exit code.",
    }
    write_json_exclusive(result_dir / "result.json", final)
    print(json.dumps({"run_dir": str(result_dir), **final}, sort_keys=True))
    return 0 if final["status"] == "completed" and not snapshot_error else 1


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
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        write_json_exclusive(args.output, freeze_payload(args.task.resolve()))
        print(args.output)
        return 0
    if args.wall_seconds < 1:
        raise ValueError("wall-seconds must be positive")
    return run_agent(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"runner: {exc}", file=sys.stderr)
        sys.exit(2)
