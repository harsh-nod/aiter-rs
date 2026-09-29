import subprocess
import unittest
from pathlib import Path

from tasks.opus_a16w16_persistent.adversarial_graph_probe import classify


SCRIPT = Path(__file__).with_name("run_adversarial_graph_host.sh")


class GraphAdmissionTests(unittest.TestCase):
    def test_noise_and_correctness_gate_parity_independently(self):
        entry = {"graph_performance": {"adapter_to_aiter_median_ratio": 1.03}}
        good = classify(entry, {"all_correct": True, "all_noise_qualified": True})
        self.assertTrue(good["noninferiority_1p05"])
        self.assertEqual(good["performance_interpretation"], "graph_feasibility_only")
        noisy = classify(entry, {"all_correct": True, "all_noise_qualified": False})
        self.assertFalse(noisy["noninferiority_1p05"])
        self.assertEqual(noisy["performance_interpretation"], "inconclusive")
        slow = classify({"graph_performance": {"adapter_to_aiter_median_ratio": 1.10}}, {"all_correct": True, "all_noise_qualified": True})
        self.assertFalse(slow["noninferiority_1p05"])

    def test_shell_watchdog_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
