import unittest

from harness.gdr_hidden_generate import build_manifest
from references.gdr_decode_packed_bf16 import (
    GuardedOutput,
    GuardedState,
    GUARD_BYTES,
    GUARD_VALUE,
    make_initial_state,
    make_step_inputs,
    oracle_step,
    validate_case,
)


class GdrOracleTests(unittest.TestCase):
    def test_duplicate_valid_slot_is_outside_contract(self):
        case = {"batch": 2, "pool": 3, "steps": 1, "seed": 7, "indices": [1, 1]}
        with self.assertRaisesRegex(ValueError, "duplicate valid"):
            validate_case(case)
        case["indices"] = [-1, -1]
        validate_case(case)

    def test_invalid_indices_write_positive_zero_and_preserve_state(self):
        import torch

        case = {"batch": 3, "pool": 2, "steps": 1, "seed": 8, "indices": [-1, 2, 2**31 - 1]}
        state = make_initial_state(case)
        before = state.clone()
        out = oracle_step(make_step_inputs(case, 0), case["indices"], state)
        self.assertTrue(torch.equal(out.view(torch.uint16), torch.zeros_like(out.view(torch.uint16))))
        self.assertTrue(torch.equal(state.view(torch.uint16), before.view(torch.uint16)))

    def test_repeated_step_rounds_state_and_keeps_other_slots(self):
        import torch

        case = {"batch": 1, "pool": 2, "steps": 2, "seed": 9, "indices": [1]}
        state = make_initial_state(case)
        untouched = state[0].clone()
        for step in range(2):
            out = oracle_step(make_step_inputs(case, step), case["indices"], state)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(state.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(state[0].view(torch.uint16), untouched.view(torch.uint16)))

    def test_generated_private_cases_satisfy_contract(self):
        generated = build_manifest()
        self.assertEqual(len(generated["cases"]), 4)
        for case in generated["cases"]:
            validate_case(case)

    def test_state_and_output_guards_detect_corruption(self):
        import torch

        storage = torch.full((GUARD_BYTES + 2 * 8 + GUARD_BYTES,), GUARD_VALUE, dtype=torch.uint8)
        guarded = GuardedState(None, storage, slot_bytes=4, stride_bytes=8, pool=2)
        guarded.check()
        storage[GUARD_BYTES + 4] = 0
        with self.assertRaisesRegex(ValueError, "slot gap"):
            guarded.check()

        output_storage = torch.full((GUARD_BYTES * 2 + 8,), GUARD_VALUE, dtype=torch.uint8)
        output = GuardedOutput(None, output_storage)
        output.check()
        output_storage[-1] = 0
        with self.assertRaisesRegex(ValueError, "output guard"):
            output.check()


if __name__ == "__main__":
    unittest.main()
