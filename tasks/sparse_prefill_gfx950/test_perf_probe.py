"""CPU-only checks for sparse-prefill graph-probe selection and statistics."""

import unittest

from tasks.sparse_prefill_gfx950.perf_probe import (
    ALTERNATING_PAIRS,
    BUCKET_CASE_IDS,
    selected_cases,
    summarize_samples,
)


class PerfProbeTests(unittest.TestCase):
    def test_buckets_are_two_distinct_public_regimes(self):
        cases = selected_cases()
        self.assertEqual(tuple(case.name for case in cases), BUCKET_CASE_IDS)
        self.assertEqual(len({case.pattern for case in cases}), 2)

    def test_thresholds_accept_stable_five_percent_gain(self):
        summary = summarize_samples([1.0] * ALTERNATING_PAIRS,
                                    [0.95] * ALTERNATING_PAIRS)
        self.assertTrue(summary["pass"])
        self.assertTrue(summary["noise_qualified"])

    def test_ratio_and_noise_are_independent_gates(self):
        slower = summarize_samples([1.0] * ALTERNATING_PAIRS,
                                   [1.06] * ALTERNATING_PAIRS)
        self.assertFalse(slower["pass"])
        self.assertTrue(slower["noise_qualified"])
        noisy = summarize_samples([1.0] * ALTERNATING_PAIRS,
                                  [0.8, 1.2] * (ALTERNATING_PAIRS // 2))
        self.assertFalse(noisy["pass"])
        self.assertFalse(noisy["noise_qualified"])

    def test_rejects_incomplete_or_invalid_samples(self):
        with self.assertRaises(ValueError):
            summarize_samples([1.0], [1.0])
        with self.assertRaises(ValueError):
            summarize_samples([1.0] * ALTERNATING_PAIRS,
                              [0.0] * ALTERNATING_PAIRS)


if __name__ == "__main__":
    unittest.main()
