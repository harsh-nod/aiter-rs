"""Public contract tests for the unscored GDR timing feasibility probe."""

from __future__ import annotations

import unittest

from harness.gdr_perf_probe import BUCKET_CASE_IDS, benchmark_cases


class GdrPerfProbeTests(unittest.TestCase):
    def test_requires_frozen_visible_one_step_buckets(self) -> None:
        cases = [
            {"id": case_id, "visibility": "visible", "steps": 1}
            for case_id in BUCKET_CASE_IDS
        ]
        self.assertEqual(benchmark_cases({"cases": cases}), cases)
        with self.assertRaisesRegex(ValueError, "missing"):
            benchmark_cases({"cases": cases[:1]})
        cases[1]["visibility"] = "withheld"
        with self.assertRaisesRegex(ValueError, "visible one-step"):
            benchmark_cases({"cases": cases})
        cases[1]["visibility"] = "visible"
        cases[1]["steps"] = 2
        with self.assertRaisesRegex(ValueError, "visible one-step"):
            benchmark_cases({"cases": cases})


if __name__ == "__main__":
    unittest.main()
