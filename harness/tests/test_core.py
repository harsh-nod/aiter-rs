import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from harness.core import merge_withheld, read_spec, score_buckets, sha256_path
from harness.run import _active_gpu_pids
from references.quant_mxfp4 import GuardedOutputs, _fp4_code, _scale_even, oracle


class CoreTests(unittest.TestCase):
    def test_bucket_rule_rejects_one_slow_shape(self):
        result = score_buckets(
            {
                "small": {"aiter_ms": [1.0, 1.01, 0.99], "candidate_ms": [1.0, 1.01, 0.99]},
                "large": {"aiter_ms": [2.0, 2.0], "candidate_ms": [2.2, 2.2]},
            },
            1.05,
        )
        self.assertFalse(result["pass"])
        self.assertTrue(result["buckets"]["small"]["pass"])
        self.assertFalse(result["buckets"]["large"]["pass"])

    def test_hash_includes_relative_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.py").write_text("x\n")
            first = sha256_path(root)
            (root / "a.py").rename(root / "b.py")
            self.assertNotEqual(first, sha256_path(root))

    def test_duplicate_cases_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            path.write_text(json.dumps({
                "schema_version": 1, "task_id": "test", "aiter_sha": "a" * 40,
                "gpu_sku": "MI350X", "target_arch": "gfx950", "plugin": "references.foo",
                "withheld_cases_sha256": "b" * 64,
                "cases": [{"id": "x", "visibility": "visible"}, {"id": "x", "visibility": "visible"}],
            }))
            with self.assertRaises(ValueError):
                read_spec(path)

    def test_withheld_manifest_requires_matching_commitment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "withheld.json"
            payload = {"schema_version": 1, "task_id": "test", "cases": [
                {"id": "secret", "visibility": "withheld"}
            ], "benchmark_case_ids": []}
            path.write_text(json.dumps(payload))
            spec = {
                "task_id": "test", "cases": [{"id": "public", "visibility": "visible"}],
                "benchmark_case_ids": [],
                "withheld_cases_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            self.assertEqual(len(merge_withheld(spec, path)["cases"]), 2)
            spec["withheld_cases_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                merge_withheld(spec, path)

    def test_contention_parser_ignores_idle_gpuagent(self):
        output = """PID PROCESS NAME GPU(s) VRAM USED SDMA USED CU OCCUPANCY
284253 gpuagent 0 0 0 0
233685 python3 1 1237176320 0 11
"""
        self.assertEqual(_active_gpu_pids(output), [233685])

    def test_even_fp4_ties_and_zero_scale(self):
        self.assertEqual(_fp4_code(0.75), 2)
        self.assertEqual(_fp4_code(1.25), 2)
        self.assertEqual(_fp4_code(-0.0), 8)
        self.assertEqual(_scale_even(0.0)[1], 0)
        self.assertEqual(_scale_even(1.75), (0.5, 126))

    def test_scalar_oracle_zero_block(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        packed, scales = oracle(torch.zeros((1, 32), dtype=torch.float16))
        self.assertEqual(packed, bytes(16))
        self.assertEqual(scales, bytes(1))

    def test_output_guard_detects_write(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        packed = torch.full((528,), 0xA5, dtype=torch.uint8)
        scales = torch.full((513,), 0xA5, dtype=torch.uint8)
        outputs = GuardedOutputs(packed[256:272], scales[256:257], packed, scales)
        outputs.check_guards()
        packed[255] = 0
        with self.assertRaisesRegex(ValueError, "packed"):
            outputs.check_guards()


if __name__ == "__main__":
    unittest.main()
