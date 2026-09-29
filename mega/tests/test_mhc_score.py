import json
import unittest
from pathlib import Path

from mega.mhc_score import GuardedTensor, output_shapes, summarize_samples


class MhcScoreTests(unittest.TestCase):
    def test_output_shapes_cover_four_contract_results(self):
        shapes = output_shapes({"m": 17, "hidden_size": 768})
        self.assertEqual(set(shapes), {"post_mix", "comb_mix", "layer_input_out", "next_residual"})
        self.assertEqual(shapes["post_mix"], ((17, 4, 1), "float32"))
        self.assertEqual(shapes["comb_mix"], ((17, 4, 4), "float32"))
        self.assertEqual(shapes["layer_input_out"], ((17, 768), "bfloat16"))
        self.assertEqual(shapes["next_residual"], ((17, 4, 768), "bfloat16"))

    def test_public_case_matrix_is_supported_by_workspace_contract(self):
        cases = json.loads((Path(__file__).parents[1] / "mhc_gfx950_cases.json").read_text())["cases"]
        for case in cases:
            m, h = case["m"], case["hidden_size"]
            expected_bytes = m * (28 + (h if case.get("norm") else 0)) * 4
            self.assertGreater(expected_bytes, 0)
            self.assertLess(expected_bytes, 256 * 1024 * 1024)

    def test_python_event_samples_cannot_claim_parity(self):
        samples = {"aiter_ms": [1.0, 1.01, 0.99], "candidate_ms": [0.9, 0.91, 0.89]}
        summary = summarize_samples(samples, graph_repetitions=0)
        self.assertFalse(summary["noise_qualified"])
        self.assertFalse(summary["exploratory_parity"])
        self.assertEqual(summary["method"], "python_event")

    def test_graph_replay_applies_frozen_noise_and_ratio_gates(self):
        samples = {"aiter_ms": [1.0, 1.01, 0.99], "candidate_ms": [1.02, 1.03, 1.01]}
        summary = summarize_samples(samples, graph_repetitions=32)
        self.assertTrue(summary["noise_qualified"])
        self.assertTrue(summary["exploratory_parity"])
        self.assertEqual(summary["graph_repetitions"], 32)
        samples["candidate_ms"] = [1.1, 1.11, 1.09]
        self.assertFalse(summarize_samples(samples, graph_repetitions=32)["exploratory_parity"])

    def test_guards_and_input_mutation_are_detected(self):
        import torch

        value = torch.arange(16, dtype=torch.float32)
        guarded = GuardedTensor((16,), torch.float32, value, device="cpu")
        self.assertTrue(guarded.guards_pass())
        self.assertTrue(guarded.input_unchanged())
        guarded.tensor[0] = -1
        self.assertFalse(guarded.input_unchanged())
        guarded.raw[guarded.start - 1] = 0
        self.assertFalse(guarded.guards_pass())


if __name__ == "__main__":
    unittest.main()
