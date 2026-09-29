"""CPU-only live public feedback tests; no Codex API, Docker, SSH, or GPU."""

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
from types import SimpleNamespace
from unittest.mock import patch

from runs.gdr_native_optimization.boundary import LIVE_TASK_DIR, sha256, validate_task
from runs.gdr_native_optimization.live_feedback import LiveFeedbackBroker, RemotePublicScorer
from runs.gdr_native_optimization.post_agent import validate_capture
from runs.runner import run_agent


def public_result(kind: str, contract: dict) -> dict:
    return {
        "status": "complete",
        "visible_case_results": {
            name: True for name in contract["visible_correctness_case_ids"]
        },
        "benchmark_bucket_results": (
            {name: {"ratio": 1.01, "pass": True, "raw_samples": [99]}
             for name in contract["benchmark_case_ids"]}
            if kind == "benchmark" else {}
        ),
        "private_note": "never agent-visible",
    }


class LiveFeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        shutil.copytree(LIVE_TASK_DIR / "starter", self.workspace)
        self.private = self.root / "trusted-feedback"
        self.task, self.contract = validate_task(LIVE_TASK_DIR)
        self.helper = LIVE_TASK_DIR / "feedback_helper.py"

    def helper_call(self, kind: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.helper), kind, "--workspace", str(self.workspace),
             "--wait-seconds", "5"],
            capture_output=True, text=True, timeout=8, check=False,
        )

    def test_public_request_round_trip_snapshots_source_and_sanitizes(self):
        calls = []

        def score(snapshot: Path, kind: str) -> dict:
            calls.append((sha256(snapshot / "kernel.hip"), kind))
            return public_result(kind, self.contract)

        broker = LiveFeedbackBroker(self.workspace, self.private, LIVE_TASK_DIR, score)
        broker.start()
        try:
            completed = self.helper_call("benchmark")
        finally:
            summary = broker.close()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        visible = json.loads(completed.stdout)
        self.assertEqual(visible["status"], "complete")
        self.assertEqual(set(visible), set(self.contract["response_fields"]))
        self.assertEqual(set(visible["visible_case_results"]), set(self.contract["visible_correctness_case_ids"]))
        self.assertEqual(set(visible["benchmark_bucket_results"]), set(self.contract["benchmark_case_ids"]))
        self.assertNotIn("raw_samples", completed.stdout)
        self.assertNotIn("never agent-visible", completed.stdout)
        self.assertEqual(calls, [(sha256(self.workspace / "kernel.hip"), "benchmark")])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(sha256(self.private / "snapshots/0001/kernel.hip"), calls[0][0])
        self.assertTrue((self.private / "raw-0001.json").is_file())
        self.assertEqual(summary["private_event_log_sha256"], sha256(self.private / "feedback_events.jsonl"))

    def test_three_request_cap_and_malformed_source_rejection(self):
        calls = []

        def score(snapshot: Path, kind: str) -> dict:
            calls.append(kind)
            return public_result(kind, self.contract)

        broker = LiveFeedbackBroker(self.workspace, self.private, LIVE_TASK_DIR, score)
        broker.start()
        try:
            for _ in range(3):
                self.assertEqual(self.helper_call("correctness").returncode, 0)
            fourth = self.helper_call("correctness")
        finally:
            summary = broker.close()
        self.assertNotEqual(fourth.returncode, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(summary["request_count"], 3)
        self.assertEqual(summary["status"], "complete")

        malformed_workspace = self.root / "malformed"
        shutil.copytree(LIVE_TASK_DIR / "starter", malformed_workspace)
        second_private = self.root / "malformed-private"
        rejected_calls = []
        rejected = LiveFeedbackBroker(
            malformed_workspace, second_private, LIVE_TASK_DIR,
            lambda *_: rejected_calls.append(True),
        )
        rejected.start()
        try:
            request = malformed_workspace / ".feedback/requests/0001.json"
            request.write_text(json.dumps({
                "schema": "aiter-rs-gdr-opt-public-request-v1", "request_id": 1,
                "source_sha256": "0" * 64, "kind": "benchmark",
            }) + "\n")
            deadline = subprocess.run(
                [sys.executable, "-c", "import time; time.sleep(.2)"], timeout=2,
                check=False,
            )
            self.assertEqual(deadline.returncode, 0)
        finally:
            rejected_summary = rejected.close()
        response = json.loads((malformed_workspace / ".feedback/responses/0001.json").read_text())
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["visible_case_results"], {})
        self.assertFalse(rejected_calls)
        self.assertEqual(rejected_summary["status"], "protocol_error")

    def test_runner_captures_fake_agent_broker_exchange_without_scoring(self):
        config = self.root / "remote-config.json"
        config.write_text("{}\n")
        fake_cli = self.root / "fake-codex"
        fake_cli.write_text("#!/bin/sh\necho codex-cli-test\n")
        fake_cli.chmod(0o700)
        fake_agent = self.root / "fake-agent.py"
        fake_agent.write_text(
            "import json, subprocess, sys\n"
            "from pathlib import Path\n"
            "workspace, helper = map(Path, sys.argv[1:3])\n"
            "sys.stdin.read()\n"
            "kernel = workspace / 'kernel.hip'\n"
            "kernel.write_text(kernel.read_text() + '\\n// fake agent edit\\n')\n"
            "reply = subprocess.run([sys.executable, str(helper), 'correctness', "
            "'--workspace', str(workspace), '--wait-seconds', '5'], "
            "capture_output=True, text=True, timeout=8)\n"
            "assert reply.returncode == 0, reply.stderr\n"
            "print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', "
            "'text': 'public response received'}}), flush=True)\n"
        )
        results = self.root / "results"
        args = argparse.Namespace(
            task=LIVE_TASK_DIR / "task.json", freeze=LIVE_TASK_DIR / "task.freeze.json",
            results=results, replicate_id="fake01", model="fake", reasoning_effort="low",
            cli=str(fake_cli), wall_seconds=15, dry_run=False, unscored_preview=True,
            feedback_config=config, feedback_private_root=self.root / "broker-private",
        )

        class FakeRemote:
            def __init__(self, *_):
                pass

            def __call__(self, snapshot: Path, kind: str) -> dict:
                return public_result(kind, self_contract)

        self_contract = self.contract
        with patch("runs.runner.agent_sandbox_command") as sandbox, patch(
            "runs.gdr_native_optimization.live_feedback.RemotePublicScorer", FakeRemote
        ):
            sandbox.side_effect = lambda cli, workspace, model, effort, helper: (
                [sys.executable, str(fake_agent), str(workspace), str(helper)],
                os.environ.copy(), {"backend": "fake-cpu"}, [],
            )
            self.assertEqual(run_agent(args), 0)
            sandbox.assert_called_once()
        run_dirs = list(results.iterdir())
        self.assertEqual(len(run_dirs), 1)
        run_dir = run_dirs[0]
        manifest = json.loads((run_dir / "manifest.json").read_text())
        result = json.loads((run_dir / "result.json").read_text())
        self.assertEqual(manifest["run_purpose"], "unscored_feedback_preview")
        self.assertFalse(manifest["incidence_eligible"])
        self.assertFalse(result["scored"])
        self.assertEqual(result["feedback"]["request_count"], 1)
        self.assertEqual(result["feedback"]["status"], "complete")
        self.assertGreaterEqual(result["snapshot_count"], 2)
        captured_task, _, final = validate_capture(run_dir, LIVE_TASK_DIR)
        self.assertEqual(captured_task["task_revision"], self.task["task_revision"])
        self.assertEqual(final["tree_sha256"], result["final_tree_sha256"])

    def test_remote_config_rejects_withheld_or_unpinned_fields(self):
        config = self.root / "bad-config.json"
        config.write_text(json.dumps({
            "schema": "aiter-rs-gdr-public-ssh-v1", "ssh_host": "mi350-2",
            "remote_root": "/tmp/public", "remote_repo": "/tmp/repo",
            "remote_aiter": "/tmp/aiter", "remote_host_gpu_report": "/tmp/gpu.json",
            "remote_repo_head": "0" * 40, "image": "pinned-image",
            "public_wall_seconds": 120, "withheld_spec": "/secret/manifest.json",
        }))
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            RemotePublicScorer(config, self.private, "safe-run")

    def test_remote_transport_pins_uploaded_source_and_returned_public_result(self):
        config = self.root / "remote-config.json"
        config.write_text(json.dumps({
            "schema": "aiter-rs-gdr-public-ssh-v1", "ssh_host": "mi350-2",
            "remote_root": "/tmp/aiter-rs-public", "remote_repo": "/tmp/trusted-repo",
            "remote_aiter": "/tmp/trusted-aiter", "remote_host_gpu_report": "/tmp/host-gpu.json",
            "remote_repo_head": "a" * 40, "image": "pinned-rocm-image",
            "public_wall_seconds": 120,
        }))
        raw = {
            "schema": "aiter-rs-gdr-opt-public-raw-v1", "status": "complete",
            "source_sha256": sha256(self.workspace / "kernel.hip"),
            "abi_header_sha256": self.task["abi_header_sha256"],
            "compiler_hipcc_sha256": self.task["compiler_hipcc_sha256"],
            "aiter_sha": self.task["aiter_sha"],
            "public_spec_sha256": self.task["harness_spec_sha256"],
            "large_fixture_sha256": self.task["large_fixture_sha256"],
            "host_gpu_report_sha256": self.task["host_gpu_report_sha256"],
            "arch": self.task["target_arch"], "gpu_name": self.task["gpu_sku"],
            **public_result("correctness", self.contract),
        }
        raw.pop("private_note")
        remote_raw = self.root / "remote-raw.json"
        remote_raw.write_text(json.dumps(raw))
        calls = []

        def remote(_, argv, timeout, stdout_path=None, stderr_path=None):
            calls.append(argv)
            if "remote-score" in argv:
                return f"AITERRS_PUBLIC_RESULT_SHA256={sha256(remote_raw)}\n"
            return ""

        def scp(argv, **kwargs):
            if argv[0] != "scp":
                raise AssertionError("unexpected host subprocess")
            if argv[-1].endswith(".json"):
                shutil.copyfile(remote_raw, argv[-1])
            return SimpleNamespace(returncode=0, stderr="")

        snapshot = self.root / "0001"
        shutil.copytree(self.workspace, snapshot)
        self.private.mkdir()
        with patch.object(RemotePublicScorer, "_remote", remote), patch(
            "runs.gdr_native_optimization.live_feedback.subprocess.run", side_effect=scp
        ):
            scorer = RemotePublicScorer(config, self.private, "test-run")
            self.assertEqual(scorer(snapshot, "correctness")["source_sha256"], raw["source_sha256"])
        self.assertTrue(any("remote-score" in argv for argv in calls))
        self.assertFalse(any("withheld" in " ".join(argv) for argv in calls))

        bad_private = self.root / "bad-private"
        bad_private.mkdir()
        raw["source_sha256"] = "0" * 64
        remote_raw.write_text(json.dumps(raw))
        with patch.object(RemotePublicScorer, "_remote", remote), patch(
            "runs.gdr_native_optimization.live_feedback.subprocess.run", side_effect=scp
        ):
            scorer = RemotePublicScorer(config, bad_private, "bad-run")
            with self.assertRaisesRegex(ValueError, "source_sha256 differs"):
                scorer(snapshot, "correctness")


if __name__ == "__main__":
    unittest.main()
