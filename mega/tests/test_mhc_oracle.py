import json
import unittest
from pathlib import Path

from mega.mhc_oracle import HC_MIXES, Parameters, make_inputs, post_pre, validate_case


CASES = json.loads((Path(__file__).parents[1] / "mhc_gfx950_cases.json").read_text())["cases"]


class MhcOracleTests(unittest.TestCase):
    def test_case_matrix_covers_fused_tails_and_fallback_boundary(self):
        ids = [case["id"] for case in CASES]
        self.assertEqual(len(ids), len(set(ids)))
        for case in CASES:
            validate_case(case)
        self.assertTrue(any(case["m"] % 16 for case in CASES if case["route"] == "fused"))
        self.assertTrue(any(case["m"] == 1024 and case["route"] == "fused" for case in CASES))
        self.assertTrue(any(case["m"] == 1025 and case["route"] == "large_m_fallback" for case in CASES))

    def test_wrong_dispatch_route_is_rejected(self):
        case = {"m": 1025, "hidden_size": 512, "seed": 1, "route": "fused"}
        with self.assertRaisesRegex(ValueError, "route does not match"):
            validate_case(case)

    def test_pretrial_native_abort_shapes_are_rejected_before_launch(self):
        excluded = (
            ({"m": 31, "hidden_size": 544, "seed": 1, "route": "fused"}, "divisible by 256"),
            ({"m": 65, "hidden_size": 512, "seed": 1, "route": "fused", "norm": True}, "RMSNorm"),
            ({"m": 1025, "hidden_size": 512, "seed": 1, "route": "large_m_fallback"}, "two residual blocks"),
        )
        for case, reason in excluded:
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, reason):
                validate_case(case)

    def test_zero_weight_identity_post_all_four_outputs(self):
        import torch

        hidden = 512
        inputs = {
            "layer_input": torch.zeros((1, hidden), dtype=torch.bfloat16),
            "residual_in": torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1)
            .expand(1, 4, hidden).to(torch.bfloat16),
            "post_layer_mix": torch.zeros((1, 4, 1), dtype=torch.float32),
            "comb_res_mix": torch.eye(4).unsqueeze(0),
            "fn": torch.zeros((HC_MIXES, 4 * hidden), dtype=torch.float32),
            "hc_scale": torch.zeros((3,), dtype=torch.float32),
            "hc_base": torch.zeros((HC_MIXES,), dtype=torch.float32),
        }
        post, comb, layer, residual = post_pre(inputs, Parameters())
        self.assertTrue(torch.equal(residual, inputs["residual_in"]))
        torch.testing.assert_close(post, torch.ones_like(post), rtol=0, atol=0)
        torch.testing.assert_close(comb, torch.full_like(comb, 0.25), rtol=0, atol=1e-5)
        self.assertTrue(torch.equal(layer, torch.full_like(layer, 5.0)))

    def test_norm_and_rank2_inputs_have_finite_four_outputs(self):
        import torch

        for case in (CASES[3], CASES[4]):
            inputs = make_inputs(case)
            post, comb, layer, residual = post_pre(inputs)
            self.assertEqual(post.shape, (case["m"], 4, 1))
            self.assertEqual(comb.shape, (case["m"], 4, 4))
            self.assertEqual(layer.shape, (case["m"], case["hidden_size"]))
            self.assertEqual(residual.shape, (case["m"], 4, case["hidden_size"]))
            self.assertTrue(all(torch.isfinite(value.float()).all() for value in (post, comb, layer, residual)))
            torch.testing.assert_close(
                comb.sum(dim=1), torch.ones_like(comb.sum(dim=1)), rtol=0, atol=2e-5,
            )


if __name__ == "__main__":
    unittest.main()
