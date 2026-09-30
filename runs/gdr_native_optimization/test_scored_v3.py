"""CPU-only regressions for the gated GDR v3 scored protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from runs.gdr_native_optimization.boundary import (
    SCORED_TASK_DIR, canonical, sha256, validate_scored_admission, validate_task,
)
from runs.gdr_native_optimization.live_feedback import (
    LiveFeedbackBroker, PublicScoreTimeout, RemotePublicScorer,
)
from runs.gdr_native_optimization.post_agent import (
    aggregate_results, capture_incidence_record, replay, validate_capture,
)
from runs.runner import digest, freeze_payload, run_agent


class ScoredV3Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task, self.contract = validate_task(SCORED_TASK_DIR)
        self.freeze = freeze_payload(SCORED_TASK_DIR / "task.json")

    def test_freeze_is_scored_but_launch_requires_separate_gpu_admission(self):
        self.assertTrue(self.task["scored_eligible"])
        self.assertEqual(self.contract["max_feedback_requests"], 3)
        self.assertEqual(self.contract["scored_protocol"]["denominator_unit"],
                         "all_launched_agent_sessions")
        self.assertEqual(self.contract["withheld_cases_sha256"], json.loads(
            (SCORED_TASK_DIR / "starter/public/spec.json").read_text()
        )["withheld_cases_sha256"])
        self.assertEqual(self.freeze, json.loads((SCORED_TASK_DIR / "task.freeze.json").read_text()))
        with self.assertRaisesRegex(ValueError, "blocked until GPU admission"):
            validate_scored_admission(SCORED_TASK_DIR, self.freeze)
        with self.assertRaisesRegex(ValueError, "unscored preview requires"):
            run_agent(argparse.Namespace(
                task=SCORED_TASK_DIR / "task.json", unscored_preview=True,
            ))

    def test_admission_checks_pinned_scorer_code_and_rejects_extra_fields(self):
        task_dir = self.root / "trusted-repo/runs/tasks/gdr_native_optimization_v3"
        task_dir.mkdir(parents=True)
        shutil.copyfile(SCORED_TASK_DIR / "task.json", task_dir / "task.json")
        receipt = {
            "schema": "aiter-rs-gdr-v3-admission-v1",
            "task_revision": self.task["task_revision"],
            "task_freeze_sha256": digest(canonical(self.freeze)),
            "aiter_sha": self.task["aiter_sha"],
            "host_gpu_report_sha256": self.task["host_gpu_report_sha256"],
            "compiler_image_id": self.task["compiler_image_id"],
            "status": "admitted",
            "public_smoke_result_sha256": "a" * 64,
            "withheld_smoke_result_sha256": "b" * 64,
        }
        receipt_path = task_dir / "admission.json"
        receipt_path.write_text(json.dumps(receipt))
        clean = CompletedProcess([], 0, "", "")
        changed = CompletedProcess([], 1, "", "")
        with patch("runs.gdr_native_optimization.boundary.SCORED_TASK_DIR", task_dir), patch(
            "runs.gdr_native_optimization.boundary.subprocess.run",
            side_effect=[clean, clean, changed, clean],
        ) as git:
            with self.assertRaisesRegex(ValueError, "trusted scorer code differs"):
                validate_scored_admission(task_dir, self.freeze)
            self.assertIn(self.task["harness_revision"], git.call_args_list[2].args[0])
        with patch("runs.gdr_native_optimization.boundary.SCORED_TASK_DIR", task_dir), patch(
            "runs.gdr_native_optimization.boundary.subprocess.run", return_value=clean,
        ):
            self.assertEqual(validate_scored_admission(task_dir, self.freeze), receipt)
            receipt["private_path"] = "/not-for-publication"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "unexpected or missing fields"):
                validate_scored_admission(task_dir, self.freeze)

    def _fake_run(self, replicate: str, exit_code: int) -> tuple[int, Path]:
        fake_cli = self.root / "fake-codex"
        fake_cli.write_text("#!/bin/sh\necho fake-codex-v1\n")
        fake_cli.chmod(0o700)
        fake_agent = self.root / "fake-agent.py"
        fake_agent.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.stdin.read()\n"
            "workspace = Path(sys.argv[1])\n"
            "with (workspace / 'kernel.hip').open('a') as out: out.write('\\n// fake edit\\n')\n"
            "raise SystemExit(int(sys.argv[2]))\n"
        )
        config = self.root / "remote-config.json"
        config.write_text("{}\n")
        results = self.root / "results"
        args = argparse.Namespace(
            task=SCORED_TASK_DIR / "task.json", freeze=SCORED_TASK_DIR / "task.freeze.json",
            results=results, replicate_id=replicate, model="fake", reasoning_effort="low",
            cli=str(fake_cli), wall_seconds=10, dry_run=False, unscored_preview=False,
            feedback_config=config, feedback_private_root=self.root / "broker-private",
        )

        class FakeRemote:
            def __init__(self, *_):
                pass

            def __call__(self, *_):
                raise AssertionError("no public request expected")

        with patch("runs.gdr_native_optimization.boundary.validate_scored_admission"), patch(
            "runs.runner.agent_sandbox_command"
        ) as sandbox, patch("runs.gdr_native_optimization.live_feedback.RemotePublicScorer", FakeRemote):
            sandbox.side_effect = lambda cli, workspace, model, effort, helper: (
                [sys.executable, str(fake_agent), str(workspace), str(exit_code)],
                os.environ.copy(), {"backend": "fake-cpu"}, [],
            )
            status = run_agent(args)
        run_id = f"{self.task['task_id']}--{self.task['task_revision']}--{replicate}"
        return status, results / run_id

    def test_scored_capture_no_feedback_and_failed_exit_keep_denominator(self):
        status, run_dir = self._fake_run("no-feedback", 0)
        self.assertEqual(status, 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        result = json.loads((run_dir / "result.json").read_text())
        self.assertEqual(manifest["run_purpose"], "scored_trial_capture")
        self.assertTrue(manifest["incidence_eligible"])
        self.assertTrue(result["incidence_eligible"])
        self.assertFalse(result["scored"])
        self.assertEqual(result["feedback"]["request_count"], 0)
        self.assertTrue(capture_incidence_record(run_dir)["replay_eligible"])
        self.assertEqual((run_dir / "workspace").stat().st_mode & 0o777, 0o700)
        self.assertEqual(run_dir.stat().st_mode & 0o777, 0o700)
        self.assertEqual(run_dir.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual((run_dir / "agent-launch.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / "broker-private").stat().st_mode & 0o777, 0o700)
        validate_capture(run_dir, SCORED_TASK_DIR)

        status, failed_dir = self._fake_run("nonzero-exit", 17)
        self.assertEqual(status, 1)
        failed_manifest = json.loads((failed_dir / "manifest.json").read_text())
        failed_result = json.loads((failed_dir / "result.json").read_text())
        self.assertTrue(failed_manifest["incidence_eligible"])
        self.assertTrue(failed_result["incidence_eligible"])
        self.assertEqual(failed_result["status"], "agent_failed")
        failed_incidence = capture_incidence_record(failed_dir)
        self.assertTrue(failed_incidence["incidence_eligible"])
        self.assertFalse(failed_incidence["replay_eligible"])
        with self.assertRaisesRegex(ValueError, "did not finish cleanly"):
            validate_capture(failed_dir, SCORED_TASK_DIR)

        interrupted = self.root / "interrupted"
        shutil.copytree(failed_dir, interrupted)
        (interrupted / "result.json").unlink()
        self.assertEqual(capture_incidence_record(interrupted)["capture_status"], "interrupted")
        self.assertTrue(capture_incidence_record(interrupted)["incidence_eligible"])
        prelaunch = self.root / "prelaunch"
        shutil.copytree(failed_dir, prelaunch)
        (prelaunch / "agent-launch.json").unlink()
        self.assertFalse(capture_incidence_record(prelaunch)["incidence_eligible"])

        result["feedback"]["status"] = "candidate_timeout_or_environment_ambiguous"
        result["feedback"]["broker_error_classes"] = ["PublicScoreTimeout"]
        (run_dir / "result.json").write_text(json.dumps(result))
        validate_capture(run_dir, SCORED_TASK_DIR)

    def test_public_feedback_timeout_is_ambiguous_and_stays_private(self):
        workspace = self.root / "workspace"
        shutil.copytree(SCORED_TASK_DIR / "starter", workspace)
        private = self.root / "private"
        broker = LiveFeedbackBroker(
            workspace, private, SCORED_TASK_DIR,
            lambda *_: (_ for _ in ()).throw(PublicScoreTimeout("watchdog")),
        )
        broker.start()
        try:
            call = subprocess.run(
                [sys.executable, str(SCORED_TASK_DIR / "feedback_helper.py"), "correctness",
                 "--workspace", str(workspace), "--wait-seconds", "5"],
                capture_output=True, text=True, timeout=8, check=False,
            )
        finally:
            summary = broker.close()
        self.assertNotEqual(call.returncode, 0)
        self.assertEqual(summary["status"], "candidate_timeout_or_environment_ambiguous")
        self.assertEqual(summary["broker_error_classes"], ["PublicScoreTimeout"])
        response = json.loads((workspace / ".feedback/responses/0001.json").read_text())
        self.assertEqual(response["status"], "error")
        self.assertNotIn("watchdog", json.dumps(response))
        self.assertEqual(json.loads((private / "raw-0001.json").read_text())["broker_error_class"],
                         "PublicScoreTimeout")

        second_workspace = self.root / "second-workspace"
        shutil.copytree(SCORED_TASK_DIR / "starter", second_workspace)
        second = LiveFeedbackBroker(
            second_workspace, self.root / "second-private", SCORED_TASK_DIR,
            lambda *_: {"status": "candidate_timeout_or_environment_ambiguous",
                        "visible_case_results": {}, "benchmark_bucket_results": {}},
        )
        second.start()
        try:
            subprocess.run(
                [sys.executable, str(SCORED_TASK_DIR / "feedback_helper.py"), "correctness",
                 "--workspace", str(second_workspace), "--wait-seconds", "5"],
                capture_output=True, text=True, timeout=8, check=False,
            )
        finally:
            second_summary = second.close()
        self.assertEqual(second_summary["status"], "candidate_timeout_or_environment_ambiguous")

    def test_remote_watchdog_marker_keeps_timeout_distinct(self):
        config = self.root / "remote-config.json"
        config.write_text(json.dumps({
            "schema": "aiter-rs-gdr-public-ssh-v1", "ssh_host": "mi350-2",
            "remote_root": "/tmp/public", "remote_repo": "/tmp/repo",
            "remote_aiter": "/tmp/aiter", "remote_host_gpu_report": "/tmp/gpu.json",
            "remote_repo_head": "a" * 40, "image": "pinned-image",
            "public_wall_seconds": 120,
        }))
        with patch.object(RemotePublicScorer, "_remote", return_value=""):
            scorer = RemotePublicScorer(config, self.root / "private", "v3-test", SCORED_TASK_DIR)
        with patch("runs.gdr_native_optimization.live_feedback.subprocess.run",
                   return_value=CompletedProcess([], 1, "AITERRS_PUBLIC_TIMEOUT\n", "private log")):
            with self.assertRaises(PublicScoreTimeout):
                scorer._remote(["test"], timeout=2)

    def _score_fixture(self, ratio: float = 1.01) -> tuple[dict, dict, dict]:
        provenance = {
            "task": self.task, "run_id": "v3-test", "task_freeze_sha256": digest(canonical(self.freeze)),
            "snapshot_sequence": 2, "final_tree_sha256": "f" * 64, "source_sha256": "s" * 64,
            "withheld_cases_sha256": self.contract["withheld_cases_sha256"],
            "capture_feedback_status": "complete",
        }
        public = {
            "status": "complete", "scored_eligible": False,
            "visible_case_results": {name: True for name in self.contract["visible_correctness_case_ids"]},
            "source_sha256": provenance["source_sha256"], "aiter_sha": self.task["aiter_sha"],
            "public_spec_sha256": self.task["harness_spec_sha256"],
            "abi_header_sha256": self.task["abi_header_sha256"],
            "large_fixture_sha256": self.task["large_fixture_sha256"],
            "compiler_hipcc_sha256": self.task["compiler_hipcc_sha256"],
            "candidate_binary_sha256": "b" * 64,
            "host_gpu_report_sha256": self.task["host_gpu_report_sha256"],
            "contention_preflight": {"self_pid_visible": True, "foreign_active_count": 0},
            "contention_postflight": {"self_pid_visible": True, "foreign_active_count": 0},
            "benchmark_bucket_results": {name: {"ratio": ratio, "pass": ratio <= 1.05}
                                         for name in self.contract["benchmark_case_ids"]},
            "raw_performance": {"buckets": {name: {"noise_qualified": True}
                                            for name in self.contract["benchmark_case_ids"]}},
            "proposed_geomean_ratio": ratio,
        }
        visible_hidden = ["valid_slots", "invalid_sentinels", "strided_mixed", "repeated_state"]
        cases = ([{"id": name, "visibility": "visible", "pass": True} for name in visible_hidden] +
                 [{"id": f"private-{i}", "visibility": "withheld", "pass": True} for i in range(4)])
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
            "correctness": {"status": "pass", "case_count": 8, "cases": cases},
        }
        return public, hidden, provenance

    def test_scored_parity_improvement_and_noise_are_distinct(self):
        public, hidden, provenance = self._score_fixture()
        summary = aggregate_results(public, hidden, provenance, 4, "p" * 64, "h" * 64, "b" * 64)
        self.assertEqual(summary["status"], "scored_replay_complete")
        self.assertTrue(summary["joint_parity_pass"])
        self.assertFalse(summary["proposed_improvement_pass"])
        self.assertTrue(summary["incidence_eligible"])
        public["proposed_geomean_ratio"] = 0.94
        self.assertTrue(aggregate_results(public, hidden, provenance, 4, "p" * 64,
                                          "h" * 64, "b" * 64)["proposed_improvement_pass"])
        public["raw_performance"]["buckets"]["valid_slots"]["noise_qualified"] = False
        invalid = aggregate_results(public, hidden, provenance, 4, "p" * 64, "h" * 64, "b" * 64)
        self.assertEqual(invalid["status"], "infrastructure_invalid")
        self.assertIsNone(invalid["joint_parity_pass"])
        self.assertTrue(invalid["retry_required"])

    def test_post_agent_public_timeout_requires_adjudication_not_retry(self):
        repo = SCORED_TASK_DIR.parents[2]
        provenance = {
            "task": self.task, "run_id": "v3-timeout", "task_freeze_sha256": digest(canonical(self.freeze)),
            "final_tree_sha256": "f" * 64, "source_sha256": "s" * 64,
            "restored_source": self.root / "source", "capture_feedback_status": "complete",
        }
        args = argparse.Namespace(
            run_dir=self.root / "run", task_dir=SCORED_TASK_DIR, repo=repo,
            aiter_source=self.root / "aiter", output=self.root / "private-result",
            withheld_spec=self.root / "withheld.json", host_gpu_report=self.root / "host.json",
            image="pinned-image", public_wall_seconds=5, hidden_wall_seconds=5,
        )

        def prepare(*_):
            args.output.mkdir(mode=0o700)
            return provenance

        with patch("runs.gdr_native_optimization.post_agent.validate_checkouts"), patch(
            "runs.gdr_native_optimization.post_agent.validate_private_inputs", return_value=4
        ), patch("runs.gdr_native_optimization.post_agent.prepare_snapshot", side_effect=prepare), patch(
            "runs.gdr_native_optimization.post_agent.public_container_command", return_value=["docker", "run"]
        ), patch("runs.gdr_native_optimization.post_agent._run_owned_container",
                 return_value={"status": "timeout"}):
            summary = replay(args)
        self.assertEqual(summary["status"], "candidate_timeout_or_environment_ambiguous")
        self.assertEqual(summary["reason"], "public_timeout")
        self.assertTrue(summary["adjudication_required"])
        self.assertIsNone(summary["retry_required"])
        self.assertTrue(summary["incidence_eligible"])
        self.assertIsNone(summary["joint_parity_pass"])

    def test_public_candidate_failure_and_setup_ambiguity_are_distinct(self):
        repo = SCORED_TASK_DIR.parents[2]
        for status, expected, score_valid, retry in (
            ("compile_failed", "scored_public_failure", True, False),
            ("candidate_load_failed", "scored_public_failure", True, False),
            ("candidate_timeout_or_environment_ambiguous",
             "candidate_timeout_or_environment_ambiguous", False, None),
            ("setup_error", "candidate_or_environment_ambiguous", False, None),
        ):
            with self.subTest(status=status):
                output = self.root / f"result-{status}"
                public, _, provenance = self._score_fixture()
                provenance["restored_source"] = self.root / "source"
                public["status"] = status
                public["visible_case_results"] = {}
                args = argparse.Namespace(
                    run_dir=self.root / "run", task_dir=SCORED_TASK_DIR, repo=repo,
                    aiter_source=self.root / "aiter", output=output,
                    withheld_spec=self.root / "withheld.json", host_gpu_report=self.root / "host.json",
                    image="pinned-image", public_wall_seconds=5, hidden_wall_seconds=5,
                )

                def prepare(*_):
                    output.mkdir(mode=0o700)
                    return provenance

                def score_stage(*_):
                    (output / "public/result.json").write_text(json.dumps(public))
                    return {"status": "failed"}

                with patch("runs.gdr_native_optimization.post_agent.validate_checkouts"), patch(
                    "runs.gdr_native_optimization.post_agent.validate_private_inputs", return_value=4
                ), patch("runs.gdr_native_optimization.post_agent.prepare_snapshot", side_effect=prepare), patch(
                    "runs.gdr_native_optimization.post_agent.public_container_command", return_value=["docker", "run"]
                ), patch("runs.gdr_native_optimization.post_agent._run_owned_container", side_effect=score_stage):
                    summary = replay(args)
                self.assertEqual(summary["status"], expected)
                self.assertIs(summary["score_valid"], score_valid)
                self.assertIs(summary["retry_required"], retry)
                self.assertTrue(summary["incidence_eligible"])

    def test_post_agent_withheld_timeout_requires_adjudication(self):
        repo = SCORED_TASK_DIR.parents[2]
        public, _, provenance = self._score_fixture()
        source = self.root / "source"
        source.mkdir()
        provenance["restored_source"] = source
        args = argparse.Namespace(
            run_dir=self.root / "run", task_dir=SCORED_TASK_DIR, repo=repo,
            aiter_source=self.root / "aiter", output=self.root / "private-result",
            withheld_spec=self.root / "withheld.json", host_gpu_report=self.root / "host.json",
            image="pinned-image", public_wall_seconds=5, hidden_wall_seconds=5,
        )

        def prepare(*_):
            args.output.mkdir(mode=0o700)
            return provenance

        def score_stage(*_):
            if not (args.output / "public/result.json").exists():
                public_dir = args.output / "public"
                (public_dir / "libcandidate.so").write_bytes(b"fake binary")
                public["candidate_binary_sha256"] = sha256(public_dir / "libcandidate.so")
                (public_dir / "result.json").write_text(json.dumps(public))
                return {"status": "completed"}
            return {"status": "timeout"}

        with patch("runs.gdr_native_optimization.post_agent.validate_checkouts"), patch(
            "runs.gdr_native_optimization.post_agent.validate_private_inputs", return_value=4
        ), patch("runs.gdr_native_optimization.post_agent.prepare_snapshot", side_effect=prepare), patch(
            "runs.gdr_native_optimization.post_agent.public_container_command", return_value=["docker", "run"]
        ), patch("runs.gdr_native_optimization.post_agent.hidden_container_command", return_value=["docker", "run"]
        ), patch("runs.gdr_native_optimization.post_agent._run_owned_container", side_effect=score_stage):
            summary = replay(args)
        self.assertEqual(summary["status"], "candidate_timeout_or_environment_ambiguous")
        self.assertEqual(summary["reason"], "withheld_timeout")
        self.assertTrue(summary["adjudication_required"])
        self.assertIsNone(summary["retry_required"])
        self.assertTrue(summary["incidence_eligible"])


if __name__ == "__main__":
    unittest.main()
