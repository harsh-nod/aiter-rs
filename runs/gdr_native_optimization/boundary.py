"""GDR optimization task freeze and public-only feedback boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable


TASK_DIR = Path(__file__).resolve().parents[1] / "tasks/gdr_native_optimization_v1"
LIVE_TASK_DIR = Path(__file__).resolve().parents[1] / "tasks/gdr_native_optimization_v2"
SCORED_TASK_DIR = Path(__file__).resolve().parents[1] / "tasks/gdr_native_optimization_v3"
SOURCE_FILES = {
    "kernel.hip",
    "gdr_decode_packed_bf16_abi.h",
    "public/spec.json",
    "public/large_fixture.json",
}
BUILD_FLAGS = ["-O3", "-shared", "-fPIC", "--offload-arch=gfx950"]
LIVE_CONTAINER_EXECUTION = {
    "uid_policy": "host_uid_nonroot",
    "nss": "minimal_readonly",
    "home": "/tmp",
    "xdg_cache_home": "/tmp/.cache",
    "aiter_jit_dir": "/tmp/aiter-jit-cache",
    "network": "none",
    "pid_namespace": "host",
}
RESPONSE_SCHEMA = "aiter-rs-gdr-opt-public-response-v1"
REQUEST_SCHEMA = "aiter-rs-gdr-opt-public-request-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def starter_hash(root: Path) -> str:
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("starter symlink is forbidden")
        if path.is_file():
            items.append([str(path.relative_to(root)), sha256(path)])
    return hashlib.sha256(canonical(items)).hexdigest()


def freeze_payload(task_dir: Path = TASK_DIR) -> dict:
    task_path = task_dir / "task.json"
    task = _read(task_path)
    payload = {
        "schema": "aiter-rs-task-freeze-v1",
        "task_sha256": sha256(task_path),
        "prompt_sha256": sha256(task_dir / task["prompt_file"]),
        "starter_tree_sha256": starter_hash(task_dir / task["starter_dir"]),
        "task_id": task["task_id"],
        "task_revision": task["task_revision"],
        "harness_revision": task["harness_revision"],
        "aiter_sha": task["aiter_sha"],
        "gpu_sku": task["gpu_sku"],
        "target_arch": task["target_arch"],
        "task_mode": task["task_mode"],
        "harness_spec_sha256": task["harness_spec_sha256"],
        "build_sources": task["build_sources"],
        "build_flags": task["build_flags"],
    }
    if task["task_mode"] == "brokered_public_feedback":
        helper = task_dir / task["feedback_helper_file"]
        payload["feedback_helper_sha256"] = sha256(helper)
    return payload


def validate_task(task_dir: Path = TASK_DIR) -> tuple[dict, dict]:
    task = _read(task_dir / "task.json")
    contract = _read(task_dir / "public_feedback.json")
    if _read(task_dir / "task.freeze.json") != freeze_payload(task_dir):
        raise ValueError("task/prompt/starter differs from frozen snapshot")
    starter = task_dir / "starter"
    scored = False
    if task["task_mode"] == "no_feedback":
        if task["scored_eligible"] is not False:
            raise ValueError("prototype must remain unscored")
    elif task["task_mode"] == "brokered_public_feedback":
        if task["task_revision"] not in {
            "native-opt-live-feedback-prototype-v2", "native-opt-live-feedback-scored-v3"
        }:
            raise ValueError("unknown live broker revision")
        scored = task["task_revision"] == "native-opt-live-feedback-scored-v3"
        if task["scored_eligible"] is not scored:
            raise ValueError("live broker eligibility differs from task revision")
        if task.get("container_execution") != LIVE_CONTAINER_EXECUTION:
            raise ValueError("live scorer container execution differs from the freeze")
        helper = task_dir / task.get("feedback_helper_file", "")
        if not helper.is_file() or sha256(helper) != task.get("feedback_helper_sha256"):
            raise ValueError("agent-visible feedback helper differs from the freeze")
        activation = "live_broker_scored_v3" if scored else "live_broker_unscored_v2"
        if contract.get("activation") != activation:
            raise ValueError("live feedback contract has the wrong activation")
        if scored:
            expected_protocol = {
                "denominator_unit": "all_launched_agent_sessions",
                "score_source": "final_immutable_snapshot",
                "private_scoring_phase": "post_agent_only",
                "hidden_confidentiality": "nonadversarial_same_process",
                "parity_max_latency_ratio_each": 1.05,
                "max_relative_mad_each_side": 0.05,
                "improvement_geomean_ratio": 0.95,
            }
            if contract.get("scored_protocol") != expected_protocol:
                raise ValueError("scored GDR protocol differs from the freeze")
            if contract.get("max_feedback_requests") != 3:
                raise ValueError("scored GDR feedback cap differs from the freeze")
    else:
        raise ValueError("unknown GDR task mode")
    if task["visible_checks"] or task["hidden_checks"]:
        raise ValueError("GDR feedback uses the trusted broker, not workspace checks")
    if task["build_sources"] != ["kernel.hip"] or task["build_flags"] != BUILD_FLAGS:
        raise ValueError("build command differs from fixed direct hipcc build")
    if task["aiter_sha"] != contract["aiter_sha"] or task["target_arch"] != "gfx950":
        raise ValueError("AITER or target architecture mismatch")
    expected = {
        "kernel.hip": task["starter_source_sha256"],
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    }
    validate_source_tree(starter, expected)
    if sha256(task_dir / "public_feedback.json") != task["feedback_contract_sha256"]:
        raise ValueError("public feedback contract changed")
    if contract["public_spec_sha256"] != task["harness_spec_sha256"] or (
        contract["large_fixture_sha256"] != task["large_fixture_sha256"]
    ):
        raise ValueError("public case commitment mismatch")
    spec = _read(starter / "public/spec.json")
    if scored:
        if contract.get("withheld_cases_sha256") != spec["withheld_cases_sha256"]:
            raise ValueError("withheld matrix commitment differs from public spec")
        limits = contract["benchmark_protocol"]
        if (limits.get("max_latency_ratio_each") != 1.05 or
                limits.get("max_relative_mad_each_side") != 0.05 or
                limits.get("proposed_geomean_improvement_ratio") != 0.95):
            raise ValueError("scored benchmark limits differ from the declared objectives")
    fixture = _read(starter / "public/large_fixture.json")
    ids = [case["id"] for case in spec["cases"]] + [fixture["case"]["id"], "default_stream_zero"]
    if ids != contract["visible_correctness_case_ids"] or not set(
        contract["benchmark_case_ids"]
    ).issubset(ids):
        raise ValueError("public case IDs differ from contract")
    if fixture["case"]["visibility"] != "visible":
        raise ValueError("large fixture is not public")
    return task, contract


def validate_scored_admission(task_dir: Path, freeze: dict) -> dict:
    """Require a separately reviewed, committed exact-v3 GPU admission receipt."""
    if task_dir.resolve() != SCORED_TASK_DIR:
        raise ValueError("scored GDR admission is only defined for v3")
    receipt_path = task_dir / "admission.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("v3 scored launch is blocked until GPU admission is committed")
    repo = task_dir.parents[2]
    relative = receipt_path.relative_to(repo).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True, text=True, timeout=15, check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", relative],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if tracked.returncode != 0 or dirty.returncode != 0 or dirty.stdout.strip():
        raise ValueError("v3 scored admission receipt must be committed and clean")
    trusted_scopes = ["harness", "references", "runs/runner.py", "runs/gdr_native_optimization"]
    revision = _read(task_dir / "task.json")["harness_revision"]
    committed_diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", revision, "HEAD", "--", *trusted_scopes],
        capture_output=True, text=True, timeout=30, check=False,
    )
    worktree = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all", "--", *trusted_scopes],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if committed_diff.returncode != 0 or worktree.returncode != 0 or worktree.stdout.strip():
        raise ValueError("v3 trusted scorer code differs from its pinned clean revision")
    receipt = _read(receipt_path)
    expected = {
        "schema": "aiter-rs-gdr-v3-admission-v1",
        "task_revision": "native-opt-live-feedback-scored-v3",
        "task_freeze_sha256": hashlib.sha256(canonical(freeze)).hexdigest(),
        "aiter_sha": _read(task_dir / "task.json")["aiter_sha"],
        "host_gpu_report_sha256": _read(task_dir / "task.json")["host_gpu_report_sha256"],
        "compiler_image_id": _read(task_dir / "task.json")["compiler_image_id"],
        "status": "admitted",
    }
    if set(receipt) != set(expected) | {"public_smoke_result_sha256", "withheld_smoke_result_sha256"}:
        raise ValueError("v3 GPU admission receipt has unexpected or missing fields")
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("v3 GPU admission receipt differs from the frozen task")
    for key in ("public_smoke_result_sha256", "withheld_smoke_result_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key, ""))):
            raise ValueError("v3 GPU admission lacks reviewed public and withheld result hashes")
    return receipt


def validate_source_tree(root: Path, expected_fixed: dict[str, str] | None = None) -> dict[str, str]:
    if root.is_symlink():
        raise ValueError("source root symlink is forbidden")
    root = root.resolve()
    found = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source symlink is forbidden")
        if path.relative_to(root).parts[0] == ".feedback":
            continue
        if path.is_file():
            relative = str(path.relative_to(root))
            if relative not in SOURCE_FILES:
                raise ValueError(f"unapproved source or build script: {relative}")
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("source file exceeds size cap")
            found[relative] = sha256(path)
    if set(found) != SOURCE_FILES:
        raise ValueError("source snapshot is incomplete")
    for name, digest in (expected_fixed or {}).items():
        if found[name] != digest:
            raise ValueError(f"pinned source changed: {name}")
    return found


def compile_argv(snapshot: Path, output: Path, task: dict) -> list[str]:
    validate_source_tree(snapshot, {
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    })
    return ["hipcc", *BUILD_FLAGS, str(snapshot / "kernel.hip"), "-o", str(output)]


def public_container_command(snapshot: Path, repo: Path, aiter: Path, output: Path,
                             host_gpu_report: Path, task: dict, image: str,
                             kind: str = "benchmark", task_dir: Path | None = None) -> list[str]:
    """Construct a public-only Docker invocation; no withheld path is accepted."""
    if kind not in {"correctness", "benchmark"}:
        raise ValueError("unknown public feedback kind")
    snapshot, repo, aiter, output = (path.resolve() for path in (snapshot, repo, aiter, output))
    if task_dir is None:
        relative_task = "runs/tasks/gdr_native_optimization_v1"
    else:
        task_dir = task_dir.resolve()
        if not task_dir.is_relative_to(repo / "runs/tasks") or task_dir.parent != repo / "runs/tasks":
            raise ValueError("task directory must be inside the trusted checkout")
        relative_task = task_dir.relative_to(repo).as_posix()
    host_gpu_report = host_gpu_report.resolve()
    if not host_gpu_report.is_file() or sha256(host_gpu_report) != task["host_gpu_report_sha256"]:
        raise ValueError("trusted host GPU report differs from task pin")
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("public output must be a fresh empty host directory")
    if output.stat().st_mode & 0o077:
        raise ValueError("public output must not be group/world accessible")
    if output.is_relative_to(snapshot) or output.is_relative_to(repo) or output.is_relative_to(aiter):
        raise ValueError("public output must be outside source mounts")
    if snapshot.is_relative_to(repo) or snapshot.is_relative_to(aiter):
        raise ValueError("snapshot must be isolated from trusted checkouts")
    validate_source_tree(snapshot, {
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    })
    observed_image_id = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    if observed_image_id != task["compiler_image_id"]:
        raise ValueError("compiler container image ID differs from task pin")
    mounts = (
        (snapshot, "/workspace/snapshot", True),
        (repo, "/workspace/aiter-rs", True),
        (aiter, "/workspace/aiter", True),
        (host_gpu_report, "/workspace/attestation/gpu.json", True),
        (output, "/workspace/output", False),
    )
    command = [
        "docker", "run", "--rm", "--network=none", "--pid=host", "--device=/dev/kfd",
        "--device=/dev/dri", "--group-add", "video", "--group-add", "render",
        "--entrypoint", "python3", "--workdir", "/workspace/aiter-rs",
    ]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    command += [
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/workspace/aiter-rs:/workspace/aiter",
        image, "-m", "runs.gdr_native_optimization.public_score",
        "--snapshot", "/workspace/snapshot", "--aiter-source", "/workspace/aiter",
        "--task-dir", f"/workspace/aiter-rs/{relative_task}",
        "--output", "/workspace/output", "--kind", kind,
        "--host-gpu-report", "/workspace/attestation/gpu.json",
    ]
    return command


def sanitize_feedback(raw: dict, request_id: int, source_sha: str,
                      raw_sha: str, kind: str, contract: dict) -> dict:
    allowed_cases = set(contract["visible_correctness_case_ids"])
    allowed_buckets = set(contract["benchmark_case_ids"])
    cases = raw.get("visible_case_results", {})
    buckets = raw.get("benchmark_bucket_results", {})
    if not isinstance(cases, dict) or not set(cases).issubset(allowed_cases):
        raise ValueError("raw scorer returned a nonpublic case ID")
    if not isinstance(buckets, dict) or not set(buckets).issubset(allowed_buckets):
        raise ValueError("raw scorer returned a nonpublic bucket ID")
    if any(type(value) is not bool for value in cases.values()):
        raise ValueError("public case result must be boolean")
    if raw.get("status") == "complete":
        if set(cases) != allowed_cases:
            raise ValueError("complete feedback lacks a public correctness case")
        if set(buckets) != (allowed_buckets if kind == "benchmark" else set()):
            raise ValueError("complete feedback has wrong public benchmark set")
    clean_buckets = {}
    for name, value in buckets.items():
        if not isinstance(value, dict) or type(value.get("pass")) is not bool:
            raise ValueError("invalid public bucket result")
        ratio = value.get("ratio")
        if not isinstance(ratio, (int, float)) or not 0 < ratio < float("inf"):
            raise ValueError("invalid public performance ratio")
        clean_buckets[name] = {"pass": value["pass"], "ratio": float(ratio)}
    return {
        "schema": RESPONSE_SCHEMA,
        "request_id": request_id,
        "source_sha256": source_sha,
        "status": raw.get("status") if raw.get("status") in {
            "complete", "correctness_failed", "compile_failed", "candidate_load_failed",
            "candidate_timeout_or_environment_ambiguous", "candidate_load_or_environment_ambiguous",
            "environment_invalid",
        } else "error",
        "visible_case_results": cases,
        "benchmark_bucket_results": clean_buckets,
        "raw_result_sha256": raw_sha,
    }


def process_request(workspace: Path, private_results: Path, request_id: int,
                    score_public: Callable[[Path, str], dict],
                    task_dir: Path = TASK_DIR) -> dict:
    """Prototype host broker: snapshot source, score public-only, return allowlisted feedback."""
    task, contract = validate_task(task_dir)
    workspace, private_results = workspace.resolve(), private_results.resolve()
    if private_results.is_relative_to(workspace) or workspace.is_relative_to(private_results):
        raise ValueError("private results and agent workspace must be disjoint")
    if private_results.is_relative_to(task_dir.resolve().parents[2]):
        raise ValueError("private feedback results must be outside the repository")
    if not 1 <= request_id <= contract["max_feedback_requests"]:
        raise ValueError("feedback request limit exceeded")
    request = workspace / ".feedback/requests" / f"{request_id:04d}.json"
    if request.is_symlink() or not request.is_file() or not request.resolve().is_relative_to(workspace):
        raise ValueError("missing or unsafe feedback request")
    data = _read(request)
    source_hashes = validate_source_tree(workspace, {
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    })
    if data != {
        "schema": REQUEST_SCHEMA, "request_id": request_id,
        "source_sha256": source_hashes["kernel.hip"], "kind": data.get("kind"),
    } or not isinstance(data["kind"], str) or data["kind"] not in {"correctness", "benchmark"}:
        raise ValueError("feedback request differs from current source or schema")
    private_results.mkdir(mode=0o700, parents=True, exist_ok=True)
    if private_results.stat().st_mode & 0o077:
        raise ValueError("private feedback directory must not be group/world accessible")
    snapshot = private_results / "snapshots" / f"{request_id:04d}"
    snapshot.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in sorted(SOURCE_FILES):
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / name, destination)
        destination.chmod(0o400)
    if validate_source_tree(snapshot) != source_hashes:
        raise ValueError("immutable snapshot differs from requested source")
    try:
        raw = score_public(snapshot, data["kind"])
    except Exception as exc:
        raw = {
            "status": "error", "error": repr(exc), "broker_error_class": type(exc).__name__,
            "visible_case_results": {}, "benchmark_bucket_results": {},
        }
    if validate_source_tree(snapshot) != source_hashes:
        raise ValueError("trusted scorer changed immutable source snapshot")
    private_raw = private_results / f"raw-{request_id:04d}.json"
    descriptor = os.open(private_raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(raw, output, sort_keys=True, indent=2)
        output.write("\n")
    response = sanitize_feedback(
        raw, request_id, source_hashes["kernel.hip"], sha256(private_raw), data["kind"], contract
    )
    responses = workspace / ".feedback/responses"
    responses.mkdir(parents=True, exist_ok=True)
    if responses.is_symlink() or not responses.resolve().is_relative_to(workspace):
        raise ValueError("unsafe response directory")
    target = responses / f"{request_id:04d}.json"
    with target.open("x", encoding="utf-8") as output:
        json.dump(response, output, sort_keys=True, indent=2)
        output.write("\n")
    event = {
        "request_id": request_id,
        "request_sha256": sha256(request),
        "source_sha256": source_hashes["kernel.hip"],
        "snapshot_tree_sha256": starter_hash(snapshot),
        "response_sha256": sha256(target),
        "raw_result_sha256": sha256(private_raw),
    }
    with (private_results / "feedback_events.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, sort_keys=True) + "\n")
    return response
