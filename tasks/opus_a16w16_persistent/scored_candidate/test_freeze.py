"""CPU-only checks for the OPUS full-K task boundary."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .freeze import HERE, clean_jit_paths, performance_gate, private_matrix, sha256, validate_matrix, validate_task, write_private


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = json.loads((HERE / "public_matrix.json").read_text())
        cls.task = json.loads((HERE / "task.json").read_text())

    def test_public_task_freeze(self):
        validate_task(self.task, self.public)
        self.assertEqual(self.task["withheld_generator_sha256"], sha256(HERE / "freeze.py"))

    def test_partial_k_and_unsupported_stride_rejected(self):
        for edit in ({"k": 194}, {"n": 2177}, {"stride_a": 256}):
            matrix = copy.deepcopy(self.public)
            matrix["cases"][0].update(edit)
            with self.assertRaises(ValueError):
                validate_matrix(matrix, withheld=False)

    def test_shape_kid_seed_and_eligibility_drift_rejected(self):
        for edit in ({"kid": 300}, {"seed": 123}):
            matrix = copy.deepcopy(self.public)
            matrix["cases"][0].update(edit)
            with self.assertRaises(ValueError):
                validate_matrix(matrix, withheld=False)
        task = copy.deepcopy(self.task)
        task["scored_eligible"] = True
        with self.assertRaises(ValueError):
            validate_task(task, self.public)

    def test_private_matrix_is_off_repo_and_full_k(self):
        private = private_matrix(self.public)
        validate_matrix(private, withheld=True)
        self.assertEqual(len(private["cases"]), 12)
        self.assertNotIn(194, [case["k"] for case in private["cases"]])
        with self.assertRaises(ValueError):
            write_private(HERE / "do-not-create.json", private)
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "secret" / "withheld.json"
            commitment = write_private(target, private)
            self.assertEqual(len(commitment), 64)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(target.parent.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                write_private(target, private)

    def test_private_public_id_and_seed_reuse_rejected(self):
        private = private_matrix(self.public)
        private["cases"][0]["id"] = self.public["cases"][0]["id"]
        with self.assertRaises(ValueError):
            validate_matrix(private, withheld=True)
        private = private_matrix(self.public)
        private["cases"][0]["seed"] = self.public["cases"][0]["seed"]
        with self.assertRaises(ValueError):
            validate_matrix(private, withheld=True)
        private = private_matrix(self.public)
        private["cases"][-1]["pattern"] = "cancellation"
        with self.assertRaises(ValueError):
            validate_matrix(private, withheld=True)

    def test_clean_jit_and_overlay_paths(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            clean_jit_paths(base / "source", base / "base-jit", base / "candidate-jit", base / "overlay")
            with self.assertRaises(ValueError):
                clean_jit_paths(base / "source", base / "base-jit", base / "base-jit", base / "overlay")
            with self.assertRaises(ValueError):
                clean_jit_paths(base / "source", base / "base-jit", base / "base-jit/sub", base / "overlay")
            (base / "base-jit").mkdir()
            with self.assertRaises(ValueError):
                clean_jit_paths(base / "source", base / "base-jit", base / "candidate-jit", base / "overlay")

    def test_performance_is_noninferiority_plus_separate_improvement(self):
        rows = [
            {"id": case["id"], "candidate_to_aiter_ratio": 1.0,
             "aiter_relative_mad": 0.01, "candidate_relative_mad": 0.01,
             "correctness_pass": True, "post_graph_readonly_pass": True,
             "exact_dispatch": True, "same_boundary": True, "pid_gate": True}
            for case in self.public["cases"]
        ]
        result = performance_gate(rows)
        self.assertTrue(result["per_bucket_noninferior"])
        self.assertFalse(result["separate_improvement_target_met"])
        rows[0]["post_graph_readonly_pass"] = False
        self.assertFalse(performance_gate(rows)["per_bucket_noninferior"])
        rows[0]["post_graph_readonly_pass"] = True
        rows[0]["candidate_to_aiter_ratio"] = 1.051
        self.assertFalse(performance_gate(rows)["per_bucket_noninferior"])


if __name__ == "__main__":
    unittest.main()
