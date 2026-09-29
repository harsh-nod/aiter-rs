import json
import unittest
from pathlib import Path

import torch

from tasks.opus_a16w16_persistent.adversarial_matrix import (
    compare_output,
    fp32_reference,
    load_matrix,
    make_operands,
    validate_case,
    validate_matrix,
)

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "adversarial_matrix_candidate.json"
ANCHORS = HERE / "cases.json"


class AdversarialMatrixTests(unittest.TestCase):
    def test_public_matrix_stays_in_exact_adapter_domain(self):
        spec = load_matrix(MATRIX, ANCHORS)
        self.assertEqual(len(spec["cases"]), 9)
        self.assertEqual({case["kid"] for case in spec["cases"]}, {300, 1300})
        self.assertEqual(len([case for case in spec["cases"] if case["source"] == "proposed"]), 6)
        self.assertEqual(validate_case(spec["cases"][-1])["grid_y_padded"], 8)

    def test_mutating_an_admitted_anchor_is_rejected(self):
        spec = json.loads(MATRIX.read_text())
        anchors = json.loads(ANCHORS.read_text())
        spec["cases"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "anchor content changed"):
            validate_matrix(spec, anchors)

    def test_unsupported_domain_is_rejected_before_gpu_call(self):
        case = {"id": "probe", "batch": 1, "m": 8192, "n": 4096, "k": 256,
                "kid": 300, "seed": 1, "pattern": "random", "source": "proposed"}
        mutations = (
            ({"n": 2177}, "16-aligned"),
            ({"k": 130}, "even count"),
            ({"m": 4096}, "two persistent"),
            ({"kid": 1300, "m": 8191}, "full M/N/K tiles"),
            ({"batch": 2}, "batch-one"),
        )
        for changes, reason in mutations:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, reason):
                validate_case(case | changes)

    def test_strided_layout_is_outside_standalone_abi(self):
        spec = json.loads(MATRIX.read_text())
        spec["physical_layout"] = "padded_A_rows"
        with self.assertRaisesRegex(ValueError, "adapter ABI"):
            validate_matrix(spec, json.loads(ANCHORS.read_text()))
        case = spec["cases"][0] | {"a_stride": 512}
        with self.assertRaisesRegex(ValueError, "public contiguous ABI"):
            validate_case(case)

    def test_private_inputs_cannot_be_inlined(self):
        spec = json.loads(MATRIX.read_text())
        spec["hidden_cases"] = [{"id": "not-public"}]
        with self.assertRaisesRegex(ValueError, "hidden data was inlined"):
            validate_matrix(spec, json.loads(ANCHORS.read_text()))

    def test_torch_oracle_phases_and_physical_b_layout(self):
        for pattern, expected_zero_phase in (
            ("checkerboard", 1), ("cancellation", 0), ("last_k_onehot", 1),
        ):
            case = {"m": 2, "n": 3, "k": 4, "pattern": pattern, "seed": 7}
            first_a, first_b = make_operands(torch, case, device="cpu", phase=0)
            second_a, second_b = make_operands(torch, case, device="cpu", phase=1)
            self.assertEqual(first_b.shape, (1, 3, 4))
            first = fp32_reference(torch, first_a, first_b)
            second = fp32_reference(torch, second_a, second_b)
            self.assertEqual(first.shape, (1, 2, 3))
            self.assertFalse(torch.equal(first, second))
            self.assertTrue(torch.all((first, second)[expected_zero_phase] == 0))
            self.assertTrue(compare_output(torch, first.to(torch.bfloat16), first)["pass"])

    def test_oracle_rejects_nonfinite_or_wrong_output(self):
        reference = torch.ones((1, 2, 3), dtype=torch.float32)
        self.assertEqual(compare_output(torch, torch.zeros_like(reference), reference)["bad_count"], 6)
        actual = reference.clone()
        actual[0, 0, 0] = float("nan")
        self.assertEqual(compare_output(torch, actual, reference)["bad_count"], 1)


if __name__ == "__main__":
    unittest.main()
