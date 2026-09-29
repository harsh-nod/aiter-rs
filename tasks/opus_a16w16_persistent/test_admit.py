import json
import unittest
from pathlib import Path

from admit import launch_geometry, validate_spec


CASES = Path(__file__).with_name("cases.json")


class AdmissionSpecTest(unittest.TestCase):
    def test_frozen_cases_are_persistent(self):
        spec = json.loads(CASES.read_text())
        validate_spec(spec)
        self.assertEqual([launch_geometry(c["m"], c["n"])["m_per_wg"] for c in spec["cases"]], [2, 3, 2])

    def test_nooob_rejects_tail(self):
        spec = json.loads(CASES.read_text())
        spec["cases"][0]["m"] -= 1
        with self.assertRaisesRegex(ValueError, "full M/N/K tiles"):
            validate_spec(spec)

    def test_persistent_tuner_excludes_unaligned_n(self):
        spec = json.loads(CASES.read_text())
        spec["cases"][2]["n"] = 2177
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_spec(spec)

    def test_nonpersistent_shape_rejected(self):
        spec = json.loads(CASES.read_text())
        spec["cases"][0]["m"] = 4096
        with self.assertRaisesRegex(ValueError, "iterate persistent"):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
