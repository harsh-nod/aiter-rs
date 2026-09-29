"""CPU-only protocol and boundary checks for the production OPUS pair."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.opus_a16w16_persistent.scored_candidate.dual_worker import (
    HERE, Worker, git_changes, paired_samples, read_matrix, select_case,
    summarize, validate_runtime_paths,
)
from tasks.opus_a16w16_persistent.scored_candidate.aggregate import aggregate_public, aggregate_withheld
from tasks.opus_a16w16_persistent.scored_candidate.freeze import private_matrix, sha256
from tasks.opus_a16w16_persistent.scored_candidate.host_identity import write_nss


class _Fake:
    def __init__(self, value: float, events: list[str], name: str):
        self.value = value
        self.events = events
        self.name = name

    def call(self, command: str) -> dict:
        self.events.append(f"{self.name}:{command}")
        return {"us_per_call": self.value}


class DualWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = json.loads((HERE / "task.json").read_text())
        cls.public = HERE / "public_matrix.json"

    def test_matrix_commitment_and_exact_case(self):
        matrix = read_matrix(self.public, self.task, withheld=False)
        self.assertEqual(select_case(matrix, "aligned-nooob")["kid"], 1300)
        with self.assertRaises(ValueError):
            select_case(matrix, "k-partial-final-tile")
        task = copy.deepcopy(self.task)
        task["public_matrix_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            read_matrix(self.public, task, withheld=False)

    def test_fresh_disjoint_jit_and_private_output(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            private = base / "private"
            private.mkdir(mode=0o700)
            validate_runtime_paths(base / "aiter", base / "overlay", base / "jit-base", base / "jit-cand", private / "result.json")
            with self.assertRaises(ValueError):
                validate_runtime_paths(base / "aiter", base / "overlay", base / "jit-base", base / "jit-base", private / "result.json")
            (base / "jit-base").mkdir()
            with self.assertRaises(ValueError):
                validate_runtime_paths(base / "aiter", base / "overlay", base / "jit-base", base / "jit-cand", private / "result.json")

    def test_git_status_rejects_extra_changes(self):
        with tempfile.TemporaryDirectory() as root:
            tree = Path(root)
            subprocess.run(["git", "init", "-q", str(tree)], check=True)
            (tree / "a.txt").write_text("a")
            subprocess.run(["git", "-C", str(tree), "add", "a.txt"], check=True)
            self.assertEqual(git_changes(tree), ["a.txt"])

    def test_alternating_serial_pairs_and_summary(self):
        events: list[str] = []
        a, b = _Fake(10.0, events, "A"), _Fake(9.5, events, "B")
        samples = paired_samples(a, b)
        self.assertEqual(events[:4], ["A:replay", "B:replay", "B:replay", "A:replay"])
        self.assertEqual(len(events), 40)
        self.assertEqual(summarize(samples)["candidate_to_aiter_ratio"], 0.95)

    def test_fake_subprocesses_have_isolated_pids_and_private_logs(self):
        with tempfile.TemporaryDirectory() as root:
            private = Path(root)
            private.chmod(0o700)
            a = Worker(fake=True, log=private / "a.log")
            b = Worker(fake=True, log=private / "b.log")
            try:
                self.assertNotEqual(a.process.pid, b.process.pid)
                self.assertNotEqual(a.process.pid, os.getpid())
                self.assertEqual(a.call("replay", value=3.0)["us_per_call"], 3.0)
                self.assertEqual(b.call("replay", value=4.0)["us_per_call"], 4.0)
            finally:
                a.close()
                b.close()
            self.assertEqual((private / "a.log").stat().st_mode & 0o777, 0o600)
            self.assertEqual((private / "b.log").stat().st_mode & 0o777, 0o600)

    def test_minimal_nonroot_nss_and_host_mount_contract(self):
        with tempfile.TemporaryDirectory() as root:
            private = Path(root)
            private.chmod(0o700)
            passwd, group = write_nss(private, os.getuid(), os.getgid())
            self.assertIn(f"aiter-replay:x:{os.getuid()}:{os.getgid()}:", passwd.read_text())
            self.assertIn(f"aiter-replay:x:{os.getgid()}:", group.read_text())
            self.assertEqual(passwd.stat().st_mode & 0o777, 0o400)
            self.assertEqual(group.stat().st_mode & 0o777, 0o400)
            self.assertEqual(write_nss(private, os.getuid(), os.getgid()), (passwd, group))
            passwd.chmod(0o600)
            passwd.write_text("wrong")
            with self.assertRaises(ValueError):
                write_nss(private, os.getuid(), os.getgid())
        script = (HERE / "run_pair_host.sh").read_text()
        self.assertIn("container-passwd,dst=/etc/passwd,readonly", script)
        self.assertIn("container-group,dst=/etc/group,readonly", script)
        self.assertIn("HOME=/tmp", script)
        self.assertIn("XDG_CACHE_HOME=/tmp/.cache", script)
        self.assertNotIn("-v /etc/passwd:/etc/passwd", script)

    def _result(self, case: dict, *, withheld: bool = False) -> dict:
        entry = {"correctness_pass": True, "exact_dispatch": True,
                 "resolved_kid": case["kid"], "workspace_used": False,
                 "jit_module_sha256": "1" * 64,
                 "performance_input_hashes": {"a_sha256": "2" * 64, "b_sha256": "3" * 64},
                 "pre_graph": {"pass": True}, "pre_timing": {"pass": True}}
        return {
            "schema": "aiter-rs-opus-production-overlay-pair-v1",
            "kind": "unscored_production_overlay_control", "scored_eligible": False,
            "case_id": case["id"], "status": "single_withheld_case_correctness_pass" if withheld else "single_bucket_pass",
            "pass": True, "aiter_sha": self.task["aiter_sha"],
            "task_sha256": sha256(HERE / "task.json"),
            "matrix_sha256": self.task["withheld_matrix_sha256" if withheld else "public_matrix_sha256"],
            "driver_sha256": sha256(HERE / "dual_worker.py"),
            "host_gpu_report_sha256": self.task["host_gpu_report_sha256"],
            "compiler_image_id": self.task["compiler_image_id"],
            "source_check": {"baseline_header_sha256": self.task["base_header_sha256"],
                             "candidate_header_sha256": "a" * 64,
                             "overlay_changed_paths": [self.task["editable_path"]]},
            "aiter": entry, "candidate": entry, "same_boundary": True, "pid_gate": True,
            "post_graph": {"aiter": {"pass": True}, "candidate": {"pass": True}},
            "graph_performance": {"candidate_to_aiter_ratio": 1.0,
                                  "aiter_relative_mad": .01, "candidate_relative_mad": .01},
        }

    def test_aggregate_requires_all_eight_exact_public_results(self):
        cases = json.loads(self.public.read_text())["cases"]
        results = [self._result(case) for case in cases]
        gate = aggregate_public(results, self.task, "a" * 64)
        self.assertTrue(gate["per_bucket_noninferior"])
        self.assertFalse(gate["separate_improvement_target_met"])
        with self.assertRaises(ValueError):
            aggregate_public(results[:-1], self.task, "a" * 64)
        altered = copy.deepcopy(results)
        altered[0]["driver_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            aggregate_public(altered, self.task, "a" * 64)

    def test_aggregate_requires_exact_private_matrix(self):
        private = private_matrix(json.loads(self.public.read_text()))
        results = [self._result(case, withheld=True) for case in private["cases"]]
        self.assertEqual(aggregate_withheld(results, private, self.task, "a" * 64)["case_count"], 12)
        results[0]["pid_gate"] = False
        with self.assertRaises(ValueError):
            aggregate_withheld(results, private, self.task, "a" * 64)


if __name__ == "__main__":
    unittest.main()
