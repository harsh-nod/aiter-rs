#!/usr/bin/env python3
"""Export an agent snapshot and replay it with a trusted ROCm Docker scorer.

The export contains source blobs and provenance only, never the agent trace.
Run ``score`` on the GPU host outside the agent sandbox. Its output is private.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from runs.runner import (
    canonical,
    checked_git_head,
    digest,
    export_snapshot,
    file_digest,
    run_limited,
    write_json_exclusive,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _full_sha(value: str, name: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a full Git SHA")
    return value


def export_bundle(run_dir: Path, output: Path) -> dict:
    run_dir = run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    result = _read(run_dir / "result.json")
    purpose = manifest.get("run_purpose")
    if purpose not in {"unscored_preview", "scored_trial_capture"}:
        raise ValueError("only completed agent captures can be exported")
    if manifest.get("incidence_eligible") != (purpose == "scored_trial_capture"):
        raise ValueError("capture purpose and incidence eligibility disagree")
    if result.get("status") != "completed" or result.get("incidence_eligible") != manifest["incidence_eligible"]:
        raise ValueError("agent capture did not complete consistently")
    freeze = manifest["task_freeze"]
    if digest(canonical(freeze)) != manifest["task_freeze_sha256"]:
        raise ValueError("task freeze hash mismatch")
    snapshots = [json.loads(line) for line in (run_dir / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    if not snapshots or snapshots[-1]["tree_sha256"] != result.get("final_tree_sha256"):
        raise ValueError("final source tree does not match completed capture")
    final = snapshots[-1]
    output.mkdir(parents=True, exist_ok=False)
    export_snapshot(run_dir, final, output / "source")
    bundle = {
        "schema": "aiter-rs-remote-score-bundle-v1",
        "run_id": manifest["run_id"],
        "run_purpose": purpose,
        "incidence_eligible": manifest["incidence_eligible"],
        "task_freeze_sha256": manifest["task_freeze_sha256"],
        "task_freeze": freeze,
        "snapshot_sequence": final["sequence"],
        "final_tree_sha256": final["tree_sha256"],
        "source_files": final["files"],
    }
    write_json_exclusive(output / "bundle.json", bundle)
    return bundle


def validate_bundle(bundle_dir: Path) -> dict:
    bundle = _read(bundle_dir / "bundle.json")
    if bundle.get("schema") != "aiter-rs-remote-score-bundle-v1":
        raise ValueError("unknown source bundle schema")
    purpose = bundle.get("run_purpose")
    if purpose not in {"unscored_preview", "scored_trial_capture"} or bundle.get("incidence_eligible") != (purpose == "scored_trial_capture"):
        raise ValueError("invalid source bundle purpose")
    if digest(canonical(bundle["task_freeze"])) != bundle["task_freeze_sha256"]:
        raise ValueError("source bundle freeze hash mismatch")
    files = bundle["source_files"]
    if digest(canonical(files)) != bundle["final_tree_sha256"]:
        raise ValueError("source bundle tree hash mismatch")
    source = bundle_dir / "source"
    actual = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("source bundle contains a symlink")
        if path.is_file():
            actual[str(path.relative_to(source))] = file_digest(path)
    if actual != files:
        raise ValueError("source bundle files differ from snapshot")
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("source bundle contains an unsafe path")
    return bundle


def validate_inputs(args: argparse.Namespace, bundle: dict) -> dict:
    freeze = bundle["task_freeze"]
    spec = _read(args.spec)
    if file_digest(args.spec) != args.spec_sha256:
        raise ValueError("public scorer spec differs from expected SHA256")
    if freeze["aiter_sha"] != spec["aiter_sha"] or freeze["target_arch"] != spec["target_arch"]:
        raise ValueError("frozen agent target differs from scorer spec")
    if freeze["gpu_sku"] not in spec["gpu_sku"]:
        raise ValueError("frozen GPU SKU differs from scorer spec")
    if bundle["incidence_eligible"]:
        if freeze["task_id"] != spec["task_id"] or freeze["harness_revision"] != args.harness_revision:
            raise ValueError("scored capture differs from pinned scorer identity")
        if freeze["harness_spec_sha256"] != args.spec_sha256:
            raise ValueError("scored capture differs from pinned scorer spec")
    if args.withheld_spec.is_relative_to(args.harness_root) or args.withheld_spec.is_relative_to(args.bundle):
        raise ValueError("private withheld manifest must stay outside public source")
    if args.host_gpu_report.is_relative_to(args.bundle):
        raise ValueError("host GPU report must stay outside agent source")
    if file_digest(args.withheld_spec) != spec["withheld_cases_sha256"]:
        raise ValueError("private withheld manifest differs from public commitment")
    withheld = _read(args.withheld_spec)
    if withheld.get("task_id") != spec["task_id"] or not withheld.get("cases"):
        raise ValueError("private withheld manifest has wrong task identity")
    report = _read(args.host_gpu_report)
    if (report.get("gpu_name"), report.get("card_model"), report.get("arch")) != (spec["gpu_sku"], spec["gpu_pci_device_id"], spec["target_arch"]):
        raise ValueError("host GPU attestation differs from scorer spec")
    checked_git_head(args.harness_root, args.harness_revision, ["harness", "references", "runs/remote_score.py", "runs/runner.py"])
    checked_git_head(args.aiter_source, spec["aiter_sha"], ["."])
    if not args.spec.is_relative_to(args.harness_root):
        raise ValueError("public scorer spec must be inside pinned harness")
    return spec


def score_on_host(args: argparse.Namespace) -> int:
    for name in ("bundle", "harness_root", "aiter_source", "spec", "withheld_spec", "host_gpu_report", "output"):
        setattr(args, name, getattr(args, name).resolve())
    _full_sha(args.harness_revision, "harness revision")
    bundle = validate_bundle(args.bundle)
    spec = validate_inputs(args, bundle)
    if any(args.output.is_relative_to(path) for path in (args.bundle, args.harness_root, args.aiter_source)):
        raise ValueError("score output must stay outside source and pinned checkouts")
    if args.output == args.withheld_spec.parent or args.output.is_relative_to(args.withheld_spec.parent):
        raise ValueError("score output must stay outside private input directory")
    image_id = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    if image_id != args.image_id:
        raise ValueError("Docker image ID differs from pinned image")
    if args.output.exists():
        raise ValueError("score output directory already exists")
    args.output.mkdir(parents=True)
    mounts = [
        (args.bundle, "/workspace/input", True),
        (args.harness_root, "/workspace/aiter-rs", True),
        (args.aiter_source, "/workspace/aiter", True),
        (args.withheld_spec, "/workspace/private/withheld.json", True),
        (args.host_gpu_report, "/workspace/private/host-report.json", True),
        (args.output, "/workspace/output", False),
    ]
    command = ["docker", "run", "--rm", "--network=none", "--pid=host", "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video", "--entrypoint", "python3", "--workdir", "/workspace/aiter-rs"]
    for source, target, readonly in mounts:
        command.extend(["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")])
    command.extend(["-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/workspace/aiter-rs:/workspace/aiter"])
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        command.extend(["-e", f"{key}={args.gpu_index}"])
    command.extend([args.image, "-m", "runs.remote_score", "_inside", "--spec", "/workspace/aiter-rs/" + str(args.spec.relative_to(args.harness_root)), "--spec-sha256", args.spec_sha256, "--bundle", "/workspace/input", "--aiter-source", "/workspace/aiter", "--withheld-spec", "/workspace/private/withheld.json", "--host-gpu-report", "/workspace/private/host-report.json", "--output", "/workspace/output"])
    outcome = run_limited(command, args.harness_root, args.output / "docker.stdout.log", args.output / "docker.stderr.log", args.wall_seconds)
    write_json_exclusive(args.output / "deployment.json", {
        "schema": "aiter-rs-remote-deployment-v1", "run_id": bundle["run_id"], "run_purpose": bundle["run_purpose"],
        "scored": False if bundle["run_purpose"] == "unscored_preview" else None,
        "harness_revision": args.harness_revision, "aiter_sha": spec["aiter_sha"], "image_id": image_id,
        "spec_sha256": args.spec_sha256, "withheld_cases_sha256": spec["withheld_cases_sha256"],
        "host_gpu_report_sha256": file_digest(args.host_gpu_report), "gpu_index": args.gpu_index,
        "correctness_only": True, "container": outcome,
    })
    summary = _read(args.output / "summary.json") if (args.output / "summary.json").exists() else {"status": "container_failed"}
    print(json.dumps({"output": str(args.output), "run_purpose": bundle["run_purpose"], **summary}, sort_keys=True))
    return 0 if outcome["status"] == "completed" and summary.get("status") == "completed" else 1


def score_inside(args: argparse.Namespace) -> int:
    bundle = validate_bundle(args.bundle)
    freeze = bundle["task_freeze"]
    if file_digest(args.spec) != args.spec_sha256:
        raise ValueError("mounted scorer spec changed")
    source_paths = [args.bundle / "source" / source for source in freeze["build_sources"]]
    if not source_paths or not all(path.is_file() and path.resolve().is_relative_to((args.bundle / "source").resolve()) for path in source_paths):
        raise ValueError("frozen build source missing or outside bundle")
    output = args.output
    library = output / "libcandidate.so"
    command = ["hipcc", *freeze["build_flags"], *(str(path) for path in source_paths), "-o", str(library)]
    build = run_limited(command, output, output / "compile.stdout.log", output / "compile.stderr.log", 180)
    if build["status"] != "completed" or not library.is_file():
        write_json_exclusive(output / "summary.json", {"status": "compile_failed", "build": build})
        return 1
    harness_cmd = [sys.executable, "-m", "harness.run", "--spec", str(args.spec), "--candidate", str(library), "--aiter-source", str(args.aiter_source), "--withheld-spec", str(args.withheld_spec), "--host-gpu-report", str(args.host_gpu_report), "--output", str(output / "harness"), "--correctness-only", "--task-freeze-sha256", bundle["task_freeze_sha256"], "--final-tree-sha256", bundle["final_tree_sha256"]]
    if bundle["run_purpose"] == "unscored_preview":
        harness_cmd.append("--unscored-preview")
    scored = run_limited(harness_cmd, Path.cwd(), output / "scorer.stdout.log", output / "scorer.stderr.log", 600, os.environ.copy())
    result_path = output / "harness" / "result.json"
    if not result_path.is_file():
        write_json_exclusive(output / "summary.json", {"status": "scorer_failed", "scorer": scored})
        return 1
    result = _read(result_path)
    expected = {"task_id": _read(args.spec)["task_id"], "aiter_sha": freeze["aiter_sha"], "spec_sha256": args.spec_sha256, "candidate_path_sha256": file_digest(library), "task_freeze_sha256": bundle["task_freeze_sha256"], "final_tree_sha256": bundle["final_tree_sha256"], "withheld_cases_sha256": _read(args.spec)["withheld_cases_sha256"], "withheld_cases_evaluated": True}
    valid = all(result.get(key) == value for key, value in expected.items()) and result.get("environment", {}).get("host_gpu_report_sha256") == file_digest(args.host_gpu_report)
    valid = valid and result.get("candidate_kind") == ("agent_preview" if bundle["run_purpose"] == "unscored_preview" else "agent_scored")
    valid = valid and result.get("performance", {}).get("status") == "not_run"
    if bundle["run_purpose"] == "unscored_preview":
        valid = valid and result.get("scored_agent_candidate") is False and result.get("joint_pass") is False
    summary = {"status": "completed" if valid else "result_invalid", "candidate_kind": result.get("candidate_kind"), "correctness": result.get("correctness", {}).get("status"), "performance": result.get("performance", {}).get("status"), "joint_pass": result.get("joint_pass"), "binary_sha256": file_digest(library), "scorer": scored}
    write_json_exclusive(output / "summary.json", summary)
    return 0 if valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--bundle", type=Path, required=True)
    score.add_argument("--harness-root", type=Path, required=True)
    score.add_argument("--harness-revision", required=True)
    score.add_argument("--aiter-source", type=Path, required=True)
    score.add_argument("--spec", type=Path, required=True)
    score.add_argument("--spec-sha256", required=True)
    score.add_argument("--withheld-spec", type=Path, required=True)
    score.add_argument("--host-gpu-report", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--image", required=True)
    score.add_argument("--image-id", required=True)
    score.add_argument("--gpu-index", type=int, default=1)
    score.add_argument("--wall-seconds", type=int, default=900)
    inside = sub.add_parser("_inside")
    for name in ("bundle", "spec", "aiter_source", "withheld_spec", "host_gpu_report", "output"):
        inside.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    inside.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    if args.action == "export":
        bundle = export_bundle(args.run_dir, args.output)
        print(json.dumps({"output": str(args.output), "run_id": bundle["run_id"], "run_purpose": bundle["run_purpose"], "snapshot_sequence": bundle["snapshot_sequence"]}, sort_keys=True))
        return 0
    if args.action == "score":
        return score_on_host(args)
    return score_inside(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"remote_score: {exc}", file=sys.stderr)
        sys.exit(2)
