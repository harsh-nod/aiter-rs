import json
import unittest
from pathlib import Path

from tasks.opus_a16w16_persistent.k_tail_repro import fresh_cases


class KTailReproTests(unittest.TestCase):
    def test_fresh_public_cases_stay_in_host_domain(self):
        root = Path(__file__).resolve().parent
        matrix = json.loads((root / "adversarial_matrix_candidate.json").read_text())
        base = next(case for case in matrix["cases"] if case["id"] == "k-partial-final-tile")
        cases = fresh_cases(base)
        self.assertEqual([(case["k"], case["seed"]) for case in cases], [(194, 95114), (254, 95115), (256, 95116)])
        self.assertTrue(all(case["pattern"] == "random" for case in cases))


if __name__ == "__main__":
    unittest.main()
