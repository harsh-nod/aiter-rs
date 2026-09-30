"""Trusted, unscored post-agent replay for native-seed GDR previews.

Public and withheld scoring run after agent capture in separate network-disabled
containers. This does not make the existing withheld scorer safe for malicious
native code: the candidate .so shares its process with the private manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from runs.gdr_native_optimization.boundary import (
    LIVE_TASK_DIR,
    SCORED_TASK_DIR,
    TASK_DIR,
    canonical,
    public_container_command,
    sha256,
    validate_source_tree,
    validate_task,
)
from runs.runner import digest, export_snapshot, freeze_payload, run_limited


SNAPSHOT_FILES = {"kernel.hip", "gdr_decode_packed_bf16_abi.h"}
SUMMARY_SCHEMA = "aiter-rs-gdr-opt-post-agent-summary-v1"
SCORED_SUMMARY_SCHEMA = "aiter-rs-gdr-opt-scored-post-agent-summary-v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _private_path(path: Path, run_dir: Path, repo: Path, aiter: Path) -> Path:
    result = path.resolve()
    if any(result.is_relative_to(parent.resolve()) for parent in (run_dir, repo, aiter)):
        raise ValueError("private scoring path overlaps agent run or source checkout")
    return result


def _write_summary(path: Path, summary: dict) -> dict:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(summary, output, sort_keys=True, indent=2)
        output.write("\n")
    return summary


def _owned_container_state(name: str, label: str, identifier: str) -> bool:
    inspected = subprocess.run(
        ["docker", "inspect", "--format", '{{.Name}} {{index .Config.Labels "aiter-rs.gdr-replay"}}', identifier],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if inspected.returncode != 0:
        if "No such object" in inspected.stderr or "No such container" in inspected.stderr:
            return False
        raise RuntimeError("Docker inspect failed; owned container state is unknown")
    if inspected.stdout.strip() != f"/{name} {label}":
        raise RuntimeError("Docker identifier belongs to a different container; refusing cleanup")
    return True


def _nss_files(private_root: Path) -> tuple[Path, Path]:
    uid, gid = os.getuid(), os.getgid()
    contents = {
        "container-passwd": (
            "root:x:0:0:root:/root:/bin/sh\n"
            "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
            f"aiter-replay:x:{uid}:{gid}:aiter replay:/tmp:/bin/sh\n"
        ),
        "container-group": (
            "root:x:0:\n"
            "nogroup:x:65534:\n"
            f"aiter-replay:x:{gid}:\n"
        ),
    }
    for name, value in contents.items():
        path = private_root / name
        if path.is_symlink():
            raise ValueError("container NSS file cannot be a symlink")
        if path.exists():
            if path.read_text(encoding="ascii") != value:
                raise ValueError("container NSS file differs from trusted host identity")
            continue
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "w", encoding="ascii") as output:
            output.write(value)
    return private_root / "container-passwd", private_root / "container-group"


def _run_owned_container(command: list[str], stage: str, private_root: Path, repo: Path,
                         wall_seconds: int) -> dict:
    if command[:2] != ["docker", "run"] or stage not in {"public", "withheld"}:
        raise ValueError("expected a trusted Docker scorer stage")
    label = hashlib.sha256(str(private_root).encode("utf-8")).hexdigest()[:20]
    name = f"aiter-rs-gdr-{label}-{stage}"
    cidfile = private_root / f"{stage}.cid"
    if cidfile.exists():
        raise ValueError("Docker CID file already exists")
    passwd, group = _nss_files(private_root)
    owned = command[:2] + [
        "--name", name, "--cidfile", str(cidfile),
        "--label", f"aiter-rs.gdr-replay={label}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/.cache",
        "-e", "USER=aiter-replay", "-e", "LOGNAME=aiter-replay",
        "-e", "AITER_JIT_DIR=/tmp/aiter-jit-cache",
        "--mount", f"type=bind,src={passwd},dst=/etc/passwd,readonly",
        "--mount", f"type=bind,src={group},dst=/etc/group,readonly",
    ] + command[2:]
    try:
        return run_limited(owned, repo, private_root / f"{stage}.stdout.log",
                           private_root / f"{stage}.stderr.log", wall_seconds)
    finally:
        identifier = name
        if cidfile.exists():
            identifier = cidfile.read_text(encoding="ascii").strip()
            if not re.fullmatch(r"[0-9a-f]{12,64}", identifier):
                raise RuntimeError("owned Docker CID file is malformed; manual inspection required")
        if _owned_container_state(name, label, identifier):
            removed = subprocess.run(["docker", "rm", "-f", identifier],
                                     capture_output=True, text=True, timeout=30, check=False)
            if removed.returncode != 0 or _owned_container_state(name, label, identifier):
                raise RuntimeError("owned scorer container could not be removed")


def capture_incidence_record(run_dir: Path, task_dir: Path = SCORED_TASK_DIR) -> dict:
    """Count launched v3 sessions even if a scorer cannot replay their capture."""
    if task_dir.resolve() != SCORED_TASK_DIR:
        raise ValueError("incidence records are defined for the scored v3 revision")
    validate_task(task_dir)
    manifest = _read(run_dir / "manifest.json")
    freeze = freeze_payload(task_dir / "task.json")
    freeze_sha = digest(canonical(freeze))
    if (manifest.get("schema") != "aiter-rs-agent-run-v1" or
            manifest.get("run_purpose") != "scored_trial_capture" or
            manifest.get("incidence_eligible") is not True or
            manifest.get("task_freeze") != freeze or
            manifest.get("task_freeze_sha256") != freeze_sha):
        raise ValueError("v3 capture manifest differs from frozen scored protocol")
    launch_path = run_dir / "agent-launch.json"
    if not launch_path.is_file():
        return {
            "run_id": manifest["run_id"], "launched": False,
            "incidence_eligible": False, "capture_status": "prelaunch_incomplete",
            "replay_eligible": False,
        }
    launch = _read(launch_path)
    if (launch.get("schema") != "aiter-rs-agent-launch-v1" or
            launch.get("run_id") != manifest["run_id"] or
            launch.get("task_freeze_sha256") != freeze_sha or
            type(launch.get("agent_pid")) is not int or launch["agent_pid"] <= 0):
        raise ValueError("agent launch receipt differs from frozen v3 run")
    result_path = run_dir / "result.json"
    result = _read(result_path) if result_path.is_file() else None
    if result is not None and result.get("incidence_eligible") is not True:
        raise ValueError("launched v3 result incorrectly excludes incidence")
    status = result.get("status", "unknown") if result is not None else "interrupted"
    return {
        "run_id": manifest["run_id"], "launched": True,
        "incidence_eligible": True, "capture_status": status,
        "replay_eligible": bool(result and status == "completed" and
                                result.get("agent_exit_code") == 0 and
                                not result.get("snapshot_errors")),
    }


def validate_capture(run_dir: Path, task_dir: Path = TASK_DIR) -> tuple[dict, dict, dict]:
    task, _ = validate_task(task_dir)
    scored = task_dir.resolve() == SCORED_TASK_DIR
    if scored and not capture_incidence_record(run_dir, task_dir)["launched"]:
        raise ValueError("scored capture has no agent launch receipt")
    manifest = _read(run_dir / "manifest.json")
    result = _read(run_dir / "result.json")
    freeze = freeze_payload(task_dir / "task.json")
    purpose = ("scored_trial_capture" if scored else
               "unscored_feedback_preview" if task_dir.resolve() == LIVE_TASK_DIR else
               "unscored_preview")
    if manifest.get("schema") != "aiter-rs-agent-run-v1" or manifest.get("run_purpose") != purpose:
        raise ValueError("capture purpose differs from frozen task revision")
    if manifest.get("incidence_eligible") is not scored or result.get("incidence_eligible") is not scored:
        raise ValueError("capture incidence eligibility differs from task revision")
    if manifest.get("task_freeze") != freeze or manifest.get("task_freeze_sha256") != digest(canonical(freeze)):
        raise ValueError("capture task freeze differs from pinned task")
    if result.get("status") != "completed" or result.get("snapshot_errors"):
        raise ValueError("agent capture did not finish cleanly")
    if result.get("agent_exit_code") != 0 or result.get("scored") is not False:
        raise ValueError("capture exit or scoring status is inconsistent")
    if purpose in {"unscored_feedback_preview", "scored_trial_capture"}:
        feedback = result.get("feedback", {})
        accepted_statuses = ({"complete", "protocol_error", "candidate_timeout_or_environment_ambiguous"}
                             if scored else {"complete"})
        if (feedback.get("status") not in accepted_statuses or feedback.get("request_limit") != 3 or
                not 0 <= feedback.get("request_count", -1) <= 3):
            raise ValueError("live public feedback capture has an incomplete broker record")
    expected_prefix = task["task_id"] + "--" + task["task_revision"] + "--"
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].startswith(expected_prefix):
        raise ValueError("agent run ID differs from task identity")
    snapshots = [json.loads(line) for line in (run_dir / "snapshots.jsonl").read_text().splitlines()]
    if not snapshots or len(snapshots) != result.get("snapshot_count"):
        raise ValueError("snapshot journal is missing or incomplete")
    if [entry.get("sequence") for entry in snapshots] != list(range(1, len(snapshots) + 1)):
        raise ValueError("snapshot sequence is not contiguous")
    final = snapshots[-1]
    if final.get("tree_sha256") != result.get("final_tree_sha256"):
        raise ValueError("final source tree differs from completed capture")
    if set(final.get("files", {})) != SNAPSHOT_FILES:
        raise ValueError("only kernel.hip and pinned ABI may appear in final source snapshot")
    if final["files"]["gdr_decode_packed_bf16_abi.h"] != task["abi_header_sha256"]:
        raise ValueError("agent changed the immutable ABI header")
    if digest(canonical(final["files"])) != final["tree_sha256"]:
        raise ValueError("final snapshot file manifest hash mismatch")
    workspace = run_dir / "workspace"
    current = validate_source_tree(workspace, {
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    })
    if current["kernel.hip"] != final["files"]["kernel.hip"]:
        raise ValueError("workspace kernel differs from captured final snapshot")
    return task, manifest, final


def prepare_snapshot(run_dir: Path, task_dir: Path, private_root: Path) -> dict:
    task, manifest, final = validate_capture(run_dir, task_dir)
    if private_root.exists():
        raise ValueError("private replay root already exists")
    private_root.mkdir(mode=0o700, parents=True)
    captured = private_root / "captured-source"
    export_snapshot(run_dir, final, captured)
    if {str(path.relative_to(captured)) for path in captured.rglob("*") if path.is_file()} != SNAPSHOT_FILES:
        raise ValueError("exported snapshot differs from source allowlist")
    restored = private_root / "restored-public-source"
    restored.mkdir(mode=0o700)
    for name in sorted(("kernel.hip", "gdr_decode_packed_bf16_abi.h", "public/spec.json", "public/large_fixture.json")):
        source = captured / name if name == "kernel.hip" else task_dir / "starter" / name
        target = restored / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o400)
    restored_hashes = validate_source_tree(restored, {
        "kernel.hip": final["files"]["kernel.hip"],
        "gdr_decode_packed_bf16_abi.h": task["abi_header_sha256"],
        "public/spec.json": task["harness_spec_sha256"],
        "public/large_fixture.json": task["large_fixture_sha256"],
    })
    return {
        "task": task,
        "run_id": manifest["run_id"],
        "task_freeze_sha256": manifest["task_freeze_sha256"],
        "snapshot_sequence": final["sequence"],
        "final_tree_sha256": final["tree_sha256"],
        "source_sha256": restored_hashes["kernel.hip"],
        "restored_source": restored,
        "capture_feedback_status": _read(run_dir / "result.json").get("feedback", {}).get("status"),
    }


def validate_private_inputs(withheld: Path, report: Path, task: dict, run_dir: Path,
                            repo: Path, aiter: Path, task_dir: Path = TASK_DIR) -> int:
    withheld = _private_path(withheld, run_dir, repo, aiter)
    report = _private_path(report, run_dir, repo, aiter)
    if withheld == report:
        raise ValueError("host report and withheld manifest are the same file")
    spec_path = task_dir / "starter/public/spec.json"
    spec = _read(spec_path)
    if sha256(spec_path) != task["harness_spec_sha256"]:
        raise ValueError("public scorer spec differs from task pin")
    if sha256(withheld) != spec["withheld_cases_sha256"]:
        raise ValueError("withheld matrix differs from public commitment")
    if sha256(report) != task["host_gpu_report_sha256"]:
        raise ValueError("MI350X host attestation differs from task pin")
    hidden = _read(withheld)
    cases = hidden.get("cases", [])
    if hidden.get("task_id") != task["task_id"] or len(cases) != 4 or any(
        case.get("visibility") != "withheld" for case in cases
    ):
        raise ValueError("withheld manifest has unexpected identity or visibility")
    host = _read(report)
    if (host.get("gpu_name"), host.get("arch"), host.get("card_model")) != (
        task["gpu_sku"], task["target_arch"], spec["gpu_pci_device_id"]
    ):
        raise ValueError("host attestation does not identify frozen MI350X")
    return len(cases)


def validate_checkouts(repo: Path, aiter: Path, task: dict, task_dir: Path = TASK_DIR) -> None:
    def git(root: Path, *argv: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *argv],
            text=True, timeout=30,
        ).strip()

    if git(aiter, "rev-parse", "HEAD") != task["aiter_sha"]:
        raise ValueError("AITER source differs from task pin")
    if git(aiter, "status", "--porcelain", "--untracked-files=all", "--"):
        raise ValueError("AITER source is dirty")
    if git(repo, "diff", "--name-only", task["harness_revision"], "HEAD", "--", "harness", "references"):
        raise ValueError("trusted harness differs from task revision")
    if task_dir.resolve() in {LIVE_TASK_DIR, SCORED_TASK_DIR} and git(
        repo, "diff", "--name-only", task["harness_revision"], "HEAD", "--",
        "runs/runner.py", "runs/gdr_native_optimization",
    ):
        raise ValueError("trusted live-feedback scorer differs from task revision")
    if git(repo, "status", "--porcelain", "--untracked-files=all", "--", "harness", "references",
           "runs/runner.py", "runs/gdr_native_optimization", str(task_dir.relative_to(repo))):
        raise ValueError("trusted replay checkout has dirty source")


def hidden_container_command(binary: Path, repo: Path, aiter: Path, withheld: Path,
                             report: Path, output: Path, task: dict, image: str,
                             task_dir: Path | None = None) -> list[str]:
    binary, repo, aiter, withheld, report, output = (
        path.resolve() for path in (binary, repo, aiter, withheld, report, output)
    )
    if not binary.is_file() or binary.suffix != ".so":
        raise ValueError("compiled candidate library is missing")
    if not output.is_dir() or any(output.iterdir()) or output.stat().st_mode & 0o077:
        raise ValueError("hidden output must be a fresh private directory")
    if sha256(report) != task["host_gpu_report_sha256"]:
        raise ValueError("host report differs from task pin")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True, timeout=30,
    ).strip()
    if image_id != task["compiler_image_id"]:
        raise ValueError("scorer image differs from task pin")
    if task_dir is None:
        task_dir = repo / "runs/tasks/gdr_native_optimization_v1"
    task_dir = task_dir.resolve()
    if task_dir not in {repo / "runs/tasks/gdr_native_optimization_v1", repo / "runs/tasks/gdr_native_optimization_v2",
                        repo / "runs/tasks/gdr_native_optimization_v3"}:
        raise ValueError("hidden scorer task is not an admitted trusted revision")
    mounts = (
        (binary, "/workspace/candidate/libcandidate.so", True),
        (repo, "/workspace/aiter-rs", True),
        (aiter, "/workspace/aiter", True),
        (withheld, "/workspace/hidden/withheld.json", True),
        (report, "/workspace/attestation/gpu.json", True),
        (output, "/workspace/output", False),
    )
    command = [
        "docker", "run", "--rm", "--network=none", "--pid=host",
        "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
        "--group-add", "render", "--entrypoint", "python3", "--workdir", "/workspace/aiter-rs",
    ]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    command += [
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/workspace/aiter-rs:/workspace/aiter",
        image, "-m", "harness.gdr_score",
        "--spec", f"/workspace/aiter-rs/{task_dir.relative_to(repo).as_posix()}/starter/public/spec.json",
        "--candidate", "/workspace/candidate/libcandidate.so",
        "--aiter-source", "/workspace/aiter",
        "--withheld-spec", "/workspace/hidden/withheld.json",
        "--host-gpu-report", "/workspace/attestation/gpu.json",
        "--output", "/workspace/output/scorer",
    ]
    return command


def aggregate_results(public: dict, hidden: dict, provenance: dict, hidden_case_count: int,
                      public_raw_sha: str, hidden_raw_sha: str, binary_sha: str) -> dict:
    visible = public.get("visible_case_results", {})
    expected_visible = {
        "valid_slots", "invalid_sentinels", "strided_mixed", "repeated_state",
        "large_valid_slots_candidate", "default_stream_zero",
    }
    if public.get("status") != "complete" or set(visible) != expected_visible or not all(
        value is True for value in visible.values()
    ):
        raise ValueError("public correctness did not pass the frozen visible set")
    if public.get("source_sha256") != provenance["source_sha256"] or public.get("aiter_sha") != provenance["task"]["aiter_sha"]:
        raise ValueError("public result provenance mismatch")
    if (public.get("public_spec_sha256"), public.get("abi_header_sha256"),
            public.get("large_fixture_sha256"), public.get("compiler_hipcc_sha256")) != (
            provenance["task"]["harness_spec_sha256"], provenance["task"]["abi_header_sha256"],
            provenance["task"]["large_fixture_sha256"], provenance["task"]["compiler_hipcc_sha256"]):
        raise ValueError("public scorer inputs or compiler differ from task pin")
    if public.get("candidate_binary_sha256") != binary_sha:
        raise ValueError("public result binary differs from compiled candidate")
    if public.get("scored_eligible") is not False:
        raise ValueError("public scorer incorrectly claims scored eligibility")
    if public.get("host_gpu_report_sha256") != provenance["task"]["host_gpu_report_sha256"]:
        raise ValueError("public result lacks pinned MI350X attestation")
    if any(public.get(key, {}).get("self_pid_visible") is not True or public[key].get("foreign_active_count") != 0
           for key in ("contention_preflight", "contention_postflight")):
        raise ValueError("public GPU contention gate did not pass")
    entries = hidden.get("correctness", {}).get("cases", [])
    withheld_entries = [entry for entry in entries if entry.get("visibility") == "withheld"]
    if hidden.get("aiter_sha") != provenance["task"]["aiter_sha"] or hidden.get("task_id") != provenance["task"]["task_id"]:
        raise ValueError("hidden scorer identity mismatch")
    if hidden.get("public_spec_raw_sha256") != provenance["task"]["harness_spec_sha256"] or (
        hidden.get("withheld_cases_sha256") != provenance["withheld_cases_sha256"]
    ):
        raise ValueError("hidden scorer input commitments differ from task pin")
    if hidden.get("candidate_binary_raw_sha256") != binary_sha or hidden.get("withheld_cases_evaluated") is not True:
        raise ValueError("hidden scorer did not evaluate this frozen binary")
    if hidden.get("environment", {}).get("host_gpu_report_sha256") != provenance["task"]["host_gpu_report_sha256"]:
        raise ValueError("hidden scorer lacks pinned MI350X attestation")
    environment = hidden["environment"]
    if environment.get("host_gpu_name") != provenance["task"]["gpu_sku"] or (
        "gfx950" not in environment.get("gpu_arch", "")
    ) or "0x75a0" not in environment.get("rocm_product", "") or (
        environment.get("host_card_model") != "0x75a0"
    ):
        raise ValueError("hidden scorer GPU identity differs from pinned MI350X")
    if len(withheld_entries) != hidden_case_count or len(entries) != hidden_case_count + 4 or (
        hidden.get("correctness", {}).get("case_count") != len(entries)
    ):
        raise ValueError("hidden scorer case count differs from commitment")
    if len({entry.get("id") for entry in entries}) != len(entries) or any(
        entry.get("visibility") not in {"visible", "withheld"} for entry in entries
    ):
        raise ValueError("hidden scorer returned duplicate or unexpected case entries")
    if {entry.get("id") for entry in entries if entry.get("visibility") == "visible"} != {
        "valid_slots", "invalid_sentinels", "strided_mixed", "repeated_state"
    }:
        raise ValueError("hidden scorer omitted or replaced public cases")
    hidden_status = hidden.get("correctness", {}).get("status")
    if hidden_status not in {"pass", "fail", "baseline_invalid"}:
        raise ValueError("hidden scorer returned an invalid correctness status")
    if hidden_status == "pass" and not all(entry.get("pass") is True for entry in entries):
        raise ValueError("hidden scorer pass contradicts case results")
    hidden_passed = sum(entry.get("pass") is True for entry in withheld_entries)
    buckets = public.get("benchmark_bucket_results", {})
    if set(buckets) != {"valid_slots", "strided_mixed", "large_valid_slots_candidate"}:
        raise ValueError("public performance bucket set differs from freeze")
    if any(type(value.get("pass")) is not bool or not isinstance(value.get("ratio"), (int, float)) or
           not math.isfinite(value["ratio"]) or value["ratio"] <= 0 for value in buckets.values()):
        raise ValueError("public performance bucket has invalid result")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "exploratory_replay_complete",
        "scored_eligible": False,
        "incidence_eligible": False,
        "agent_parity_claim": False,
        "run_id": provenance["run_id"],
        "task_freeze_sha256": provenance["task_freeze_sha256"],
        "snapshot_sequence": provenance["snapshot_sequence"],
        "final_tree_sha256": provenance["final_tree_sha256"],
        "source_sha256": provenance["source_sha256"],
        "binary_sha256": binary_sha,
        "aiter_sha": provenance["task"]["aiter_sha"],
        "host_gpu_report_sha256": provenance["task"]["host_gpu_report_sha256"],
        "public_raw_result_sha256": public_raw_sha,
        "hidden_raw_result_sha256": hidden_raw_sha,
        "visible_passed_count": sum(value is True for value in visible.values()),
        "visible_case_count": len(visible),
        "withheld_passed_count": hidden_passed,
        "withheld_case_count": hidden_case_count,
        "withheld_correctness_status": hidden_status,
        "public_bucket_ratios": {name: round(value["ratio"], 6) for name, value in buckets.items()},
        "public_noninferiority_pass": all(value["pass"] is True for value in buckets.values()),
        "proposed_improvement_pass": public.get("proposed_geomean_ratio", 1e9) <= 0.95 and all(
            value["pass"] is True for value in buckets.values()
        ),
        "hidden_same_process_confidentiality_guarantee": False,
    }
    if provenance["task"]["scored_eligible"] is True:
        scored_buckets = public.get("raw_performance", {}).get("buckets", {})
        qualified = set(scored_buckets) == set(buckets) and all(
            entry.get("noise_qualified") is True for entry in scored_buckets.values()
        )
        score_valid = qualified and hidden_status != "baseline_invalid"
        correctness_pass = hidden_status == "pass" and all(entry.get("pass") is True for entry in entries)
        parity_pass = correctness_pass and all(value["pass"] is True for value in buckets.values())
        summary.update({
            "schema": SCORED_SUMMARY_SCHEMA,
            "status": "scored_replay_complete" if score_valid else "infrastructure_invalid",
            "candidate_kind": "agent_trial",
            "scored_eligible": True,
            "incidence_eligible": True,
            "score_valid": score_valid,
            "retry_required": not score_valid,
            "capture_feedback_status": provenance.get("capture_feedback_status"),
            "public_noise_qualified": qualified,
            "joint_parity_pass": parity_pass if score_valid else None,
            "public_noninferiority_pass": (all(value["pass"] is True for value in buckets.values())
                                            if score_valid else None),
            "proposed_improvement_pass": (parity_pass and public.get("proposed_geomean_ratio", 1e9) <= 0.95
                                          if score_valid else None),
            "agent_parity_claim": parity_pass if score_valid else False,
        })
    return summary


def replay(args) -> dict:
    run_dir, task_dir, repo, aiter, private_root = (
        path.resolve() for path in (args.run_dir, args.task_dir, args.repo, args.aiter_source, args.output)
    )
    private_root = _private_path(private_root, run_dir, repo, aiter)
    if task_dir not in {repo / "runs/tasks/gdr_native_optimization_v1", repo / "runs/tasks/gdr_native_optimization_v2",
                        repo / "runs/tasks/gdr_native_optimization_v3"}:
        raise ValueError("task directory must be the pinned trusted checkout task")
    task, _ = validate_task(task_dir)
    validate_checkouts(repo, aiter, task, task_dir)
    hidden_count = validate_private_inputs(args.withheld_spec, args.host_gpu_report, task, run_dir, repo, aiter, task_dir)
    provenance = prepare_snapshot(run_dir, task_dir, private_root)
    provenance["withheld_cases_sha256"] = _read(task_dir / "starter/public/spec.json")["withheld_cases_sha256"]
    public_dir = private_root / "public"
    public_dir.mkdir(mode=0o700)
    public_command = public_container_command(
        provenance["restored_source"], repo, aiter, public_dir, args.host_gpu_report,
        task, args.image, "benchmark", task_dir,
    )
    public_run = _run_owned_container(public_command, "public", private_root, repo,
                                      args.public_wall_seconds)
    public_result_path = public_dir / "result.json"
    if public_run["status"] == "timeout" or not public_result_path.is_file():
        if task["scored_eligible"] is True:
            return _write_summary(private_root / "summary.json", {
                "schema": SCORED_SUMMARY_SCHEMA,
                "status": "candidate_timeout_or_environment_ambiguous",
                "reason": "public_timeout" if public_run["status"] == "timeout" else "public_missing_result",
                "candidate_kind": "agent_trial",
                "scored_eligible": True,
                "incidence_eligible": True,
                "score_valid": False,
                "retry_required": None,
                "adjudication_required": True,
                "capture_feedback_status": provenance.get("capture_feedback_status"),
                "joint_parity_pass": None,
                "proposed_improvement_pass": None,
                "run_id": provenance["run_id"],
                "task_freeze_sha256": provenance["task_freeze_sha256"],
                "final_tree_sha256": provenance["final_tree_sha256"],
                "source_sha256": provenance["source_sha256"],
                "withheld_evaluated": False,
            })
        raise RuntimeError("public-only scorer did not complete; raw logs stay private")
    public = _read(public_result_path)
    public_raw_sha = sha256(public_result_path)
    public_status = public.get("status")
    if public_status in {
        "compile_failed", "correctness_failed", "candidate_load_failed",
        "candidate_timeout_or_environment_ambiguous", "candidate_load_or_environment_ambiguous",
        "environment_invalid", "setup_error",
    }:
        if public_run["status"] != "failed":
            raise RuntimeError("public scorer exit status contradicts its failure result")
        visible = public.get("visible_case_results", {})
        allowed_visible = set(_read(task_dir / "public_feedback.json")["visible_correctness_case_ids"])
        if not isinstance(visible, dict) or not set(visible).issubset(allowed_visible) or any(
            type(value) is not bool for value in visible.values()
        ):
            raise ValueError("public failure returned invalid visible case data")
        scored = task["scored_eligible"] is True
        infrastructure = public_status == "environment_invalid"
        ambiguous = public_status in {
            "candidate_timeout_or_environment_ambiguous", "candidate_load_or_environment_ambiguous",
            "setup_error",
        }
        if scored and public_status not in {"environment_invalid", "setup_error"}:
            if (public.get("source_sha256") != provenance["source_sha256"] or
                    public.get("aiter_sha") != task["aiter_sha"] or
                    public.get("public_spec_sha256") != task["harness_spec_sha256"] or
                    public.get("compiler_hipcc_sha256") != task["compiler_hipcc_sha256"]):
                raise ValueError("public failure provenance differs from frozen scored candidate")
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "public_stage_failed",
            "public_stage_status": public_status,
            "scored_eligible": False,
            "incidence_eligible": False,
            "agent_parity_claim": False,
            "run_id": provenance["run_id"],
            "task_freeze_sha256": provenance["task_freeze_sha256"],
            "snapshot_sequence": provenance["snapshot_sequence"],
            "final_tree_sha256": provenance["final_tree_sha256"],
            "source_sha256": provenance["source_sha256"],
            "pinned_aiter_sha": task["aiter_sha"],
            "pinned_host_gpu_report_sha256": task["host_gpu_report_sha256"],
            "public_raw_result_sha256": public_raw_sha,
            "visible_passed_count": sum(visible.values()),
            "visible_case_count": len(visible),
            "withheld_evaluated": False,
            "hidden_same_process_confidentiality_guarantee": False,
        }
        if scored:
            summary.update({
                "schema": SCORED_SUMMARY_SCHEMA,
                "status": ("infrastructure_invalid" if infrastructure else
                           "candidate_timeout_or_environment_ambiguous"
                           if public_status == "candidate_timeout_or_environment_ambiguous" else
                           "candidate_or_environment_ambiguous" if ambiguous else
                           "scored_public_failure"),
                "candidate_kind": "agent_trial",
                "scored_eligible": True,
                "incidence_eligible": True,
                "score_valid": not infrastructure and not ambiguous,
                "retry_required": True if infrastructure else None if ambiguous else False,
                "adjudication_required": ambiguous,
                "capture_feedback_status": provenance.get("capture_feedback_status"),
                "joint_parity_pass": None if infrastructure or ambiguous else False,
                "proposed_improvement_pass": None if infrastructure or ambiguous else False,
            })
        return _write_summary(private_root / "summary.json", summary)
    if public_run["status"] != "completed" or public_status != "complete":
        raise RuntimeError("public scorer exit status contradicts its result")
    binary = public_dir / "libcandidate.so"
    if not binary.is_file() or public.get("candidate_binary_sha256") != sha256(binary):
        raise ValueError("public scorer did not produce the pinned candidate binary")
    expected_visible = set(_read(task_dir / "public_feedback.json")["visible_correctness_case_ids"])
    if public.get("status") != "complete" or set(public.get("visible_case_results", {})) != expected_visible or not all(
        value is True for value in public["visible_case_results"].values()
    ):
        raise RuntimeError("public correctness did not pass; withheld replay was not run")
    hidden_dir = private_root / "withheld"
    hidden_dir.mkdir(mode=0o700)
    hidden_command = hidden_container_command(
        binary, repo, aiter, args.withheld_spec, args.host_gpu_report, hidden_dir,
        task, args.image, task_dir,
    )
    hidden_run = _run_owned_container(hidden_command, "withheld", private_root, repo,
                                      args.hidden_wall_seconds)
    hidden_result_path = hidden_dir / "scorer/result.json"
    if hidden_run["status"] == "timeout" or not hidden_result_path.is_file():
        if task["scored_eligible"] is True:
            return _write_summary(private_root / "summary.json", {
                "schema": SCORED_SUMMARY_SCHEMA,
                "status": "candidate_timeout_or_environment_ambiguous",
                "reason": "withheld_timeout" if hidden_run["status"] == "timeout" else "withheld_missing_result",
                "candidate_kind": "agent_trial",
                "scored_eligible": True,
                "incidence_eligible": True,
                "score_valid": False,
                "retry_required": None,
                "adjudication_required": True,
                "capture_feedback_status": provenance.get("capture_feedback_status"),
                "joint_parity_pass": None,
                "proposed_improvement_pass": None,
                "run_id": provenance["run_id"],
                "task_freeze_sha256": provenance["task_freeze_sha256"],
                "final_tree_sha256": provenance["final_tree_sha256"],
                "source_sha256": provenance["source_sha256"],
                "binary_sha256": sha256(binary),
                "public_raw_result_sha256": public_raw_sha,
                "withheld_evaluated": False,
            })
        raise RuntimeError("withheld scorer did not complete; raw logs stay private")
    hidden_result = _read(hidden_result_path)
    hidden_status = hidden_result.get("correctness", {}).get("status")
    if task["scored_eligible"] is True and hidden_status not in {"pass", "fail", "baseline_invalid"}:
        return _write_summary(private_root / "summary.json", {
            "schema": SCORED_SUMMARY_SCHEMA,
            "status": "candidate_or_environment_ambiguous",
            "reason": "withheld_scorer_invalid_status",
            "candidate_kind": "agent_trial",
            "scored_eligible": True,
            "incidence_eligible": True,
            "score_valid": False,
            "retry_required": None,
            "adjudication_required": True,
            "capture_feedback_status": provenance.get("capture_feedback_status"),
            "joint_parity_pass": None,
            "proposed_improvement_pass": None,
            "run_id": provenance["run_id"],
            "task_freeze_sha256": provenance["task_freeze_sha256"],
            "final_tree_sha256": provenance["final_tree_sha256"],
            "source_sha256": provenance["source_sha256"],
            "binary_sha256": sha256(binary),
            "public_raw_result_sha256": public_raw_sha,
            "hidden_raw_result_sha256": sha256(hidden_result_path),
            "withheld_evaluated": False,
        })
    if (hidden_run["status"] == "completed") != (hidden_status == "pass"):
        raise RuntimeError("withheld scorer exit status contradicts its result; raw logs stay private")
    summary = aggregate_results(
        public, hidden_result, provenance, hidden_count,
        public_raw_sha, sha256(hidden_result_path), sha256(binary),
    )
    return _write_summary(private_root / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--aiter-source", required=True, type=Path)
    parser.add_argument("--withheld-spec", required=True, type=Path)
    parser.add_argument("--host-gpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--public-wall-seconds", type=int, default=600)
    parser.add_argument("--hidden-wall-seconds", type=int, default=600)
    args = parser.parse_args()
    summary = replay(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
