import unittest
from types import SimpleNamespace
from unittest import mock

from harness.gdr_hidden_generate import build_manifest
from harness.gdr_score import correctness_status, run_case
from references.gdr_decode_packed_bf16 import (
    GuardedInput,
    GuardedOutput,
    GuardedState,
    GUARD_BYTES,
    GUARD_VALUE,
    HipCandidate,
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

        input_storage = torch.full((GUARD_BYTES * 2 + 8,), GUARD_VALUE, dtype=torch.uint8)
        input_guard = GuardedInput(input_storage, input_storage.clone())
        input_guard.check()
        input_storage[0] = 0
        with self.assertRaisesRegex(AssertionError, "input or input guard"):
            input_guard.check()

    def test_candidate_abi_uses_element_strides_and_supplied_stream(self):
        calls = []

        def launch(*args):
            calls.append(args)
            return 0

        class Tensor:
            def __init__(self, ptr, shape, row_stride):
                self.ptr, self.shape, self.row_stride = ptr, shape, row_stride

            def data_ptr(self):
                return self.ptr

            def stride(self, dim):
                if dim != 0:
                    raise AssertionError("only row or slot stride is expected")
                return self.row_stride

        inputs = {
            "mixed_qkv": Tensor(11, (3, 6144), 12288),
            "a": Tensor(12, (3, 32), 64),
            "b": Tensor(13, (3, 32), 64),
            "dt_bias": Tensor(14, (32,), 0),
            "A_log": Tensor(15, (32,), 0),
            "indices": Tensor(16, (3,), 2),
        }
        candidate = HipCandidate(launch)
        candidate.run(
            inputs, Tensor(17, (5, 32, 128, 128), 524416),
            Tensor(18, (3, 1, 32, 128), 4096), 99,
        )
        self.assertEqual(len(launch.argtypes), 17)
        self.assertEqual([arg.value for arg in calls[0][:15]], [
            11, 12, 13, 14, 15, 16, 17, 18,
            3, 12288, 64, 64, 2, 524416, 5,
        ])
        self.assertAlmostEqual(calls[0][15].value, 128**-0.5)
        self.assertEqual(calls[0][16].value, 99)

    def test_aiter_success_does_not_count_as_candidate_success(self):
        import torch

        class Tensor:
            def __init__(self, ptr, origin):
                self.ptr, self.origin = ptr, origin

            def data_ptr(self):
                return self.ptr

        class Plugin:
            allocations = 0

            def validate_case(self, case):
                pass

            def make_initial_state(self, case):
                return {"steps": 0}

            def guarded_state(self, state, *, slot_padding):
                self.allocations += 1
                origin = "aiter" if self.allocations == 1 else "candidate"
                return SimpleNamespace(tensor=Tensor(self.allocations, origin))

            def make_step_inputs(self, case, step):
                return {"step": step}

            def gpu_step_inputs(self, cpu_inputs, case):
                return cpu_inputs, []

            def guarded_output(self, batch):
                return SimpleNamespace(tensor=Tensor(99, "output"))

            def oracle_step(self, inputs, indices, state):
                if state["steps"] != inputs["step"]:
                    raise AssertionError("AITER state leaked into candidate oracle")
                state["steps"] += 1
                return state["steps"]

            def run_aiter(self, inputs, state, out):
                return out, state

            def compare_step(self, cpu_inputs, indices, expected_out, expected_state,
                             gpu_inputs, gpu_state, gpu_out, input_guards):
                if gpu_state.tensor.origin == "candidate":
                    raise AssertionError("candidate differs from oracle")

        candidate = SimpleNamespace(run=mock.Mock())
        case = {"id": "visible", "visibility": "visible", "batch": 1,
                "pool": 1, "steps": 2, "indices": [0]}
        with mock.patch.object(torch.cuda, "synchronize"), mock.patch.object(
            torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=123)
        ):
            result = run_case(Plugin(), candidate, case)
        self.assertTrue(result["aiter"]["pass"])
        self.assertEqual(len(result["aiter"]["steps"]), 2)
        self.assertFalse(result["candidate"]["pass"])
        self.assertEqual(len(result["candidate"]["steps"]), 1)
        self.assertEqual(correctness_status([result], False), "fail")
        candidate.run.assert_called_once()

    def test_baseline_invalid_takes_precedence(self):
        self.assertEqual(correctness_status([
            {"aiter": {"pass": False}, "candidate": {"pass": True}}
        ], True), "baseline_invalid")
        passing = [{"aiter": {"pass": True}, "candidate": {"pass": True}}]
        self.assertEqual(correctness_status(passing, False), "visible_pass")
        self.assertEqual(correctness_status(passing, True), "pass")


if __name__ == "__main__":
    unittest.main()
