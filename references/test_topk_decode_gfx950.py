"""CPU-only tests for the task-local oracle and adversarial cases."""

import unittest
import tempfile
from pathlib import Path

import torch

from references.topk_decode_admission import generate_private, load_cases
from references.topk_decode_gfx950 import Case, make_logits, oracle, public_cases, stable_topk_indices


class StableDecodeOracleTest(unittest.TestCase):
    def test_signed_zero_and_index_ties(self):
        row = torch.tensor([0.0, -0.0, 2.0, 2.0, -1.0], dtype=torch.float32)
        self.assertEqual(stable_topk_indices(row, 3), [0, 2, 3])
        self.assertEqual(stable_topk_indices(row, 6), [0, 1, 2, 3, 4, -1])

    def test_effective_decode_lengths(self):
        case = Case("small", "next_n", 1, (4, 3), next_n=2, rows=4, width=5, k=2)
        self.assertEqual([case.effective_length(row) for row in range(4)], [3, 4, 2, 3])

    def test_padding_and_poison_tail(self):
        case = Case("small", "padded", 1, (2,), rows=1, width=5, k=3)
        logits = torch.tensor([[1.0, 2.0, 100.0, 100.0, 100.0]])
        indices, values = oracle(case, logits)
        self.assertEqual(indices.tolist(), [[0, 1, -1]])
        self.assertEqual(values.tolist(), [[1.0, 2.0, float("-inf")]])

    def test_public_matrix_contract(self):
        for case in public_cases():
            case.validate()
            self.assertEqual((case.rows, case.width, case.k), (4, 65_536, 512))
            logits = make_logits(case)
            self.assertEqual(logits.dtype, torch.float32)

    def test_private_matrix_is_exclusive_and_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hidden.json"
            generate_private(path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(load_cases(path)), 5)
            with self.assertRaises(FileExistsError):
                generate_private(path)


if __name__ == "__main__":
    unittest.main()
