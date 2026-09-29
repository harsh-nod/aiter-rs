"""CPU-only trust-boundary tests; these do not run an agent or GPU scorer."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from subprocess import CompletedProcess

from runs.gdr_native_optimization.boundary import (
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    TASK_DIR,
    compile_argv,
    process_request,
    public_container_command,
    sanitize_feedback,
    sha256,
    validate_task,
)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "agent"
        shutil.copytree(TASK_DIR / "starter", self.workspace)
        self.private = self.root / "private-results"
        self.task, self.contract = validate_task()

    def write_request(self, number=1, kind="correctness"):
        directory = self.workspace / ".feedback/requests"
        directory.mkdir(parents=True, exist_ok=True)
        request = {
            "schema": REQUEST_SCHEMA, "request_id": number,
            "source_sha256": sha256(self.workspace / "kernel.hip"), "kind": kind,
        }
        (directory / f"{number:04d}.json").write_text(json.dumps(request) + "\n")

    def test_freeze_and_pinned_sources_validate(self):
        self.assertEqual(self.task["task_mode"], "no_feedback")
        self.assertFalse(self.task["scored_eligible"])
        self.assertEqual(self.task["build_sources"], ["kernel.hip"])

    def test_fixed_compiler_argv_never_executes_workspace_script(self):
        marker = self.root / "script-ran"
        script = self.workspace / "build.sh"
        script.write_text(f"#!/bin/sh\ntouch {marker}\n")
        script.chmod(0o755)
        self.write_request()
        scorer_calls = []
        with self.assertRaisesRegex(ValueError, "unapproved source or build script"):
            process_request(
                self.workspace, self.private, 1,
                lambda *_: scorer_calls.append(True),
            )
        self.assertFalse(marker.exists())
        self.assertFalse(scorer_calls)
        script.unlink()
        command = compile_argv(self.workspace, self.root / "candidate.so", self.task)
        self.assertEqual(command[0:5], ["hipcc", "-O3", "-shared", "-fPIC", "--offload-arch=gfx950"])
        self.assertEqual(command[-2:], ["-o", str(self.root / "candidate.so")])
        self.assertNotIn("build.sh", " ".join(command))

    def test_public_container_has_no_hidden_mount_or_network(self):
        snapshot = self.root / "snapshot"
        shutil.copytree(TASK_DIR / "starter", snapshot)
        repo = self.root / "trusted-repo"
        aiter = self.root / "aiter"
        output = self.root / "public-output"
        output.mkdir(mode=0o700)
        with patch("runs.gdr_native_optimization.boundary.subprocess.run") as inspect:
            inspect.return_value = CompletedProcess([], 0, self.task["compiler_image_id"] + "\n", "")
            command = public_container_command(
                snapshot, repo, aiter, output, self.task, "pinned-image",
            )
            inspect.assert_called_once()
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertEqual(len(mounts), 4)
        self.assertIn("--network=none", command)
        self.assertNotIn("--pid=host", command)
        self.assertTrue(all("withheld" not in mount for mount in mounts))
        self.assertFalse(any("/workspace/private" in mount for mount in mounts))
        self.assertEqual(
            {mount.split(",dst=")[1].split(",")[0] for mount in mounts},
            {"/workspace/snapshot", "/workspace/aiter-rs", "/workspace/aiter", "/workspace/output"},
        )
        with patch("runs.gdr_native_optimization.boundary.subprocess.run") as inspect:
            inspect.return_value = CompletedProcess([], 0, "sha256:wrong\n", "")
            with self.assertRaisesRegex(ValueError, "image ID"):
                public_container_command(snapshot, repo, aiter, output, self.task,
                                         "pinned-image")

    def test_broker_preserves_snapshot_and_only_returns_public_fields(self):
        self.write_request(kind="benchmark")

        def fake_public_scorer(snapshot, kind):
            self.assertEqual(kind, "benchmark")
            self.assertNotEqual(snapshot, self.workspace)
            self.assertEqual(sha256(snapshot / "kernel.hip"), sha256(self.workspace / "kernel.hip"))
            return {
                "status": "complete",
                "visible_case_results": {name: True for name in self.contract["visible_correctness_case_ids"]},
                "benchmark_bucket_results": {
                    name: {"ratio": 0.94, "pass": True, "raw_samples": [123]}
                    for name in self.contract["benchmark_case_ids"]
                },
                "host_paths": ["private-only-data"],
            }

        response = process_request(self.workspace, self.private, 1, fake_public_scorer)
        self.assertEqual(response["schema"], RESPONSE_SCHEMA)
        self.assertEqual(set(response), set(self.contract["response_fields"]))
        self.assertNotIn("private-only-data", json.dumps(response))
        self.assertNotIn("raw_samples", json.dumps(response))
        self.assertTrue((self.private / "snapshots/0001/kernel.hip").is_file())
        self.assertTrue((self.private / "feedback_events.jsonl").is_file())
        self.assertEqual(response["raw_result_sha256"], sha256(self.private / "raw-0001.json"))
        with self.assertRaises(FileExistsError):
            process_request(self.workspace, self.private, 1, fake_public_scorer)

    def test_hidden_case_result_is_never_sanitized_into_feedback(self):
        raw = {
            "status": "complete",
            "visible_case_results": {"hidden_case": True},
            "benchmark_bucket_results": {},
        }
        with self.assertRaisesRegex(ValueError, "nonpublic case"):
            sanitize_feedback(raw, 1, "a" * 64, "b" * 64, "correctness", self.contract)

    def test_request_source_hash_and_cap_are_enforced(self):
        self.write_request()
        (self.workspace / "kernel.hip").write_text("edited source\n")
        with self.assertRaisesRegex(ValueError, "differs from current source"):
            process_request(self.workspace, self.private, 1, lambda *_: {})
        with self.assertRaisesRegex(ValueError, "request limit"):
            process_request(self.workspace, self.private, 4, lambda *_: {})


if __name__ == "__main__":
    unittest.main()
