import unittest

from feasibility import REPEATS, summarize


class GraphSummaryTest(unittest.TestCase):
    def test_equal_stable_samples(self):
        summary = summarize({"aiter_us": [10.0] * REPEATS, "adapter_us": [10.0] * REPEATS})
        self.assertEqual(summary["adapter_to_aiter_median_ratio"], 1.0)
        self.assertTrue(summary["noise_qualified"])

    def test_noisy_candidate_is_not_qualified(self):
        summary = summarize({"aiter_us": [10.0] * REPEATS, "adapter_us": [8.0, 12.0] * (REPEATS // 2)})
        self.assertFalse(summary["noise_qualified"])


if __name__ == "__main__":
    unittest.main()
