import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("probe_partial_k.py")
spec = importlib.util.spec_from_file_location("opus_partial_k_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeStaticTests(unittest.TestCase):
    def test_cases_meet_tuner_loop_rule(self):
        probe.check_cases()
        self.assertEqual([probe.k_loop_count(k) for k in probe.K_VALUES], [4, 4, 4])
        self.assertEqual(probe.k_loop_count(192), 3)

    def test_public_controls_are_distinct(self):
        self.assertEqual(probe.K_VALUES, (194, 254, 256))
        self.assertEqual(len(set(probe.SEEDS.values())), 3)
        self.assertEqual(probe.KID, 300)
        self.assertGreater(probe.GUARD, 0)

    def test_classification_requires_oracle_and_padding_controls(self):
        def result(bad):
            return {"bad_elements": bad, "guards_intact": True, "inputs_unchanged": True, "padding_unchanged": True}

        output = {
            "contiguous": {"194": result(9), "254": result(7), "256": result(0)},
            "k194_padded_row": {"zero_padding": result(0), "one_padding": result(10), "outputs_differ": True},
        }
        self.assertEqual(probe.classify(output), (True, True))
        output["contiguous"]["256"]["bad_elements"] = 1
        self.assertEqual(probe.classify(output), (True, False))
        output["contiguous"]["256"]["bad_elements"] = 0
        output["k194_padded_row"]["one_padding"]["guards_intact"] = False
        self.assertEqual(probe.classify(output), (False, False))


if __name__ == "__main__":
    unittest.main()
