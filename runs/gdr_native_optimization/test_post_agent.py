"""CPU-only tests for the unscored GDR post-agent replay boundary."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runs.gdr_native_optimization.boundary import TASK_DIR, canonical, sha256, validate_task
from runs.gdr_native_optimization.post_agent import (
    aggregate_results,
    hidden_container_command,
    prepare_snapshot,
    replay,
    validate_capture,
    validate_private_inputs,
)
from runs.runner import Snapshotter, digest, freeze_payload


class PostAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "agent-run"
        self.workspace = self.run_dir / "workspace"
        self.run_dir.mkdir()
        shutil.copytree(TASK_DIR / "starter", self.workspace)
        self.task, self.contract = validate_task()
        self.freeze = freeze_payload(TASK_DIR / "task.json")
        self.run_id = self.task["task_id"] + "--" + self.task["task_revision"] + "--r001"
        self.write_capture()

    def write_capture(self):
        snapshotter = Snapshotter(self.workspace, self.run_dir)
        snapshotter.capture("starter")
        with (self.workspace / "kernel.hip").open("a", encoding="utf-8") as output:
            output.write("\n// agent edit\n")
        snapshotter.capture("final")
        manifest = {
            "schema": "aiter-rs-agent-run-v1", "run_id": self.run_id,
            "run_purpose": "unscored_preview", "incidence_eligible": False,
            "task_freeze": self.freeze, "task_freeze_sha256": digest(canonical(self.freeze)),
        }
        result = {
            "status": "completed", "agent_exit_code": 0, "scored": False,
            "incidence_eligible": False, "snapshot_errors": [],
            "snapshot_count": snapshotter.sequence,
            "final_tree_sha256": snapshotter.previous_tree,
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest))
        (self.run_dir / "result.json").write_text(json.dumps(result))

    def test_completed_snapshot_is_restored_with_only_agent_kernel(self):
        task, manifest, final = validate_capture(self.run_dir)
        self.assertEqual(task["task_id"], self.task["task_id"])
        self.assertEqual(manifest["run_id"], self.run_id)
        private = self.root / "private"
        provenance = prepare_snapshot(self.run_dir, TASK_DIR, private)
        restored = provenance["restored_source"]
        self.assertEqual(sha256(restored / "kernel.hip"), final["files"]["kernel.hip"])
        self.assertEqual(sha256(restored / "gdr_decode_packed_bf16_abi.h"), self.task["abi_header_sha256"])
        self.assertEqual(sha256(restored / "public/spec.json"), self.task["harness_spec_sha256"])
        self.assertEqual(sha256(restored / "public/large_fixture.json"), self.task["large_fixture_sha256"])
        self.assertEqual(provenance["snapshot_sequence"], 2)
        self.assertFalse((restored / "build.sh").exists())

    def test_capture_rejects_mutable_fixed_files_and_uncaptured_edits(self):
        changed = self.workspace / "gdr_decode_packed_bf16_abi.h"
        changed.write_text(changed.read_text() + "\n// changed\n")
        with self.assertRaisesRegex(ValueError, "pinned source changed"):
            validate_capture(self.run_dir)
        shutil.copyfile(TASK_DIR / "starter/gdr_decode_packed_bf16_abi.h", changed)
        (self.workspace / "build.sh").write_text("#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(ValueError, "unapproved source or build script"):
            validate_capture(self.run_dir)
        (self.workspace / "build.sh").unlink()
        with (self.workspace / "kernel.hip").open("a", encoding="utf-8") as output:
            output.write("// uncaptured\n")
        with self.assertRaisesRegex(ValueError, "workspace kernel differs"):
            validate_capture(self.run_dir)

    def test_capture_rejects_forged_result_and_blob(self):
        result_path = self.run_dir / "result.json"
        result = json.loads(result_path.read_text())
        result["final_tree_sha256"] = "0" * 64
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "final source tree differs"):
            validate_capture(self.run_dir)
        result["final_tree_sha256"] = json.loads((self.run_dir / "snapshots.jsonl").read_text().splitlines()[-1])["tree_sha256"]
        result_path.write_text(json.dumps(result))
        final = json.loads((self.run_dir / "snapshots.jsonl").read_text().splitlines()[-1])
        (self.run_dir / "blobs" / final["files"]["kernel.hip"]).write_text("tampered")
        with self.assertRaisesRegex(ValueError, "snapshot blob hash mismatch"):
            prepare_snapshot(self.run_dir, TASK_DIR, self.root / "private")

    def test_private_inputs_are_committed_and_outside_agent_workspace(self):
        private = self.root / "private"
        private.mkdir(mode=0o700)
        withheld = private / "withheld.json"
        report = private / "host.json"
        withheld.write_text(json.dumps({"task_id": self.task["task_id"], "cases": [
            {"visibility": "withheld"} for _ in range(4)
        ]}))
        report.write_text(json.dumps({"gpu_name": self.task["gpu_sku"], "arch": "gfx950", "card_model": "0x75a0"}))
        frozen = {**self.task, "host_gpu_report_sha256": sha256(report)}
        spec_path = TASK_DIR / "starter/public/spec.json"
        spec = json.loads(spec_path.read_text())
        self.assertNotEqual(sha256(withheld), spec["withheld_cases_sha256"])
        with self.assertRaisesRegex(ValueError, "withheld matrix differs"):
            validate_private_inputs(withheld, report, frozen, self.run_dir, TASK_DIR.parents[2], self.root / "aiter")
        with self.assertRaisesRegex(ValueError, "overlaps agent run"):
            validate_private_inputs(self.workspace / "public/spec.json", report, frozen,
                                    self.run_dir, TASK_DIR.parents[2], self.root / "aiter")

    def test_hidden_container_is_separate_networkless_and_private(self):
        binary = self.root / "libcandidate.so"
        binary.write_bytes(b"test binary")
        report = self.root / "host.json"
        report.write_bytes(b"test report")
        withheld = self.root / "withheld.json"
        withheld.write_bytes(b"private data")
        output = self.root / "hidden-output"
        output.mkdir(mode=0o700)
        task = {**self.task, "host_gpu_report_sha256": sha256(report)}
        with patch("runs.gdr_native_optimization.post_agent.subprocess.check_output", return_value=task["compiler_image_id"] + "\n"):
            command = hidden_container_command(binary, self.root / "repo", self.root / "aiter",
                                               withheld, report, output, task, "pinned-image")
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertIn("--network=none", command)
        self.assertIn("--pid=host", command)
        self.assertEqual(len(mounts), 6)
        self.assertTrue(any(f"src={withheld},dst=/workspace/hidden/withheld.json,readonly" in mount for mount in mounts))
        self.assertFalse(any("agent-run" in mount for mount in mounts))
        self.assertFalse(any("workspace/snapshot" in mount for mount in mounts))
        self.assertIn("--withheld-spec", command)

    def test_aggregate_is_sanitized_and_records_failed_hidden_correctness(self):
        provenance = {
            "task": self.task, "run_id": self.run_id, "task_freeze_sha256": digest(canonical(self.freeze)),
            "snapshot_sequence": 2, "final_tree_sha256": "f" * 64, "source_sha256": "s" * 64,
            "withheld_cases_sha256": "w" * 64,
        }
        public = {
            "status": "complete", "scored_eligible": False,
            "visible_case_results": {case: True for case in self.contract["visible_correctness_case_ids"]},
            "source_sha256": provenance["source_sha256"], "aiter_sha": self.task["aiter_sha"],
            "public_spec_sha256": self.task["harness_spec_sha256"],
            "abi_header_sha256": self.task["abi_header_sha256"],
            "large_fixture_sha256": self.task["large_fixture_sha256"],
            "compiler_hipcc_sha256": self.task["compiler_hipcc_sha256"],
            "candidate_binary_sha256": "b" * 64,
            "host_gpu_report_sha256": self.task["host_gpu_report_sha256"],
            "contention_preflight": {"self_pid_visible": True, "foreign_active_count": 0},
            "contention_postflight": {"self_pid_visible": True, "foreign_active_count": 0},
            "benchmark_bucket_results": {name: {"ratio": 0.99, "pass": True} for name in self.contract["benchmark_case_ids"]},
            "proposed_geomean_ratio": 0.99,
        }
        hidden = {
            "aiter_sha": self.task["aiter_sha"], "task_id": self.task["task_id"],
            "public_spec_raw_sha256": self.task["harness_spec_sha256"],
            "withheld_cases_sha256": provenance["withheld_cases_sha256"],
            "candidate_binary_raw_sha256": "b" * 64, "withheld_cases_evaluated": True,
            "environment": {
                "host_gpu_report_sha256": self.task["host_gpu_report_sha256"],
                "host_gpu_name": self.task["gpu_sku"], "host_card_model": "0x75a0",
                "gpu_arch": "gfx950:sramecc+:xnack-", "rocm_product": "Card Model: 0x75a0",
            },
            "correctness": {"status": "fail", "case_count": 8, "cases": [
                *({"id": name, "visibility": "visible", "pass": True} for name in self.contract["visible_correctness_case_ids"][:4]),
                *({"id": "private-case-" + str(index), "visibility": "withheld", "pass": index != 3} for index in range(4)),
            ]},
        }
        summary = aggregate_results(public, hidden, provenance, 4, "p" * 64, "h" * 64, "b" * 64)
        self.assertEqual(summary["withheld_passed_count"], 3)
        self.assertEqual(summary["withheld_correctness_status"], "fail")
        self.assertFalse(summary["agent_parity_claim"])
        self.assertFalse(summary["scored_eligible"])
        self.assertNotIn("private-case", json.dumps(summary))
        hidden["candidate_binary_raw_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "frozen binary"):
            aggregate_results(public, hidden, provenance, 4, "p" * 64, "h" * 64, "b" * 64)

    def test_fake_replay_orders_public_before_hidden_and_keeps_raw_private(self):
        output = self.root / "private-replay"
        args = SimpleNamespace(
            run_dir=self.run_dir, task_dir=TASK_DIR, repo=TASK_DIR.parents[2],
            aiter_source=self.root / "aiter", withheld_spec=self.root / "withheld.json",
            host_gpu_report=self.root / "host.json", output=output, image="pinned-image",
            public_wall_seconds=10, hidden_wall_seconds=10,
        )
        events = []

        def fake_run(command, cwd, stdout, stderr, wall):
            self.assertFalse(stdout.is_relative_to(self.workspace))
            self.assertFalse(stderr.is_relative_to(self.workspace))
            events.append(command[0])
            if command[0] == "public":
                binary = output / "public/libcandidate.so"
                binary.write_bytes(b"compiled candidate")
                (output / "public/result.json").write_text(json.dumps({
                    "status": "complete", "candidate_binary_sha256": sha256(binary),
                    "visible_case_results": {
                        case: True for case in self.contract["visible_correctness_case_ids"]
                    },
                }))
            else:
                scorer = output / "withheld/scorer"
                scorer.mkdir()
                (scorer / "result.json").write_text(json.dumps({"correctness": {"status": "pass"}}))
            return {"status": "completed"}

        with patch("runs.gdr_native_optimization.post_agent.validate_checkouts"), patch(
            "runs.gdr_native_optimization.post_agent.validate_private_inputs", return_value=4
        ), patch("runs.gdr_native_optimization.post_agent.public_container_command", return_value=["public"]), patch(
            "runs.gdr_native_optimization.post_agent.hidden_container_command", return_value=["hidden"]
        ), patch("runs.gdr_native_optimization.post_agent.run_limited", side_effect=fake_run), patch(
            "runs.gdr_native_optimization.post_agent.aggregate_results",
            return_value={"schema": "fake-summary", "withheld_passed_count": 4},
        ):
            summary = replay(args)
        self.assertEqual(events, ["public", "hidden"])
        self.assertEqual(summary["withheld_passed_count"], 4)
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        self.assertFalse((self.workspace / "withheld.json").exists())
        self.assertFalse((self.workspace / "summary.json").exists())

    def test_fake_public_failure_never_opens_hidden_stage(self):
        output = self.root / "private-replay"
        args = SimpleNamespace(
            run_dir=self.run_dir, task_dir=TASK_DIR, repo=TASK_DIR.parents[2],
            aiter_source=self.root / "aiter", withheld_spec=self.root / "withheld.json",
            host_gpu_report=self.root / "host.json", output=output, image="pinned-image",
            public_wall_seconds=10, hidden_wall_seconds=10,
        )

        def fake_run(command, cwd, stdout, stderr, wall):
            binary = output / "public/libcandidate.so"
            binary.write_bytes(b"compiled candidate")
            (output / "public/result.json").write_text(json.dumps({
                "status": "correctness_failed", "candidate_binary_sha256": sha256(binary),
                "visible_case_results": {"valid_slots": False},
            }))
            return {"status": "failed"}

        with patch("runs.gdr_native_optimization.post_agent.validate_checkouts"), patch(
            "runs.gdr_native_optimization.post_agent.validate_private_inputs", return_value=4
        ), patch("runs.gdr_native_optimization.post_agent.public_container_command", return_value=["public"]), patch(
            "runs.gdr_native_optimization.post_agent.hidden_container_command"
        ) as hidden_command, patch("runs.gdr_native_optimization.post_agent.run_limited", side_effect=fake_run):
            summary = replay(args)
            hidden_command.assert_not_called()
        self.assertEqual(summary["status"], "public_stage_failed")
        self.assertEqual(summary["public_stage_status"], "correctness_failed")
        self.assertFalse(summary["withheld_evaluated"])
        self.assertNotIn("visible_case_results", summary)


if __name__ == "__main__":
    unittest.main()
