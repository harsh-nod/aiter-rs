import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness.core import read_spec
from references import gdr_decode_packed_bf16 as gdr
from tasks.gdr_native_optimization import large_fixture_probe, profile_dispatch


SPEC = Path(gdr.__file__).with_name("gdr_decode_packed_bf16_gfx950.json")
FIXTURE = Path(large_fixture_probe.__file__).with_name("large_valid_slots_candidate.json")


class DispatchTraceTests(unittest.TestCase):
    def test_only_pinned_public_buckets_are_selectable(self):
        spec = read_spec(SPEC)
        fixture = large_fixture_probe.load_fixture(FIXTURE, spec)
        for name in profile_dispatch.BUCKETS:
            self.assertEqual(profile_dispatch.select_case(spec, fixture, name)["id"], name)
        with self.assertRaisesRegex(ValueError, "public bucket"):
            profile_dispatch.select_case(spec, fixture, "invalid_sentinels")

    def test_one_call_checks_oracle_and_guards_for_both_launch_paths(self):
        case = {"batch": 16, "indices": list(range(16)), "state_slot_padding": True}
        tensor = SimpleNamespace(data_ptr=lambda: 42)
        cpu_state = SimpleNamespace(clone=lambda: object())
        state = SimpleNamespace(tensor=tensor)
        output = SimpleNamespace(tensor=tensor)
        plugin = SimpleNamespace(
            make_initial_state=lambda unused: cpu_state,
            make_step_inputs=lambda unused, step: {"x": step},
            gpu_step_inputs=lambda unused, case: ({"x": "gpu"}, ["guard"]),
            guarded_state=lambda unused, slot_padding: state,
            guarded_output=lambda unused: output,
            oracle_step=lambda unused, indices, expected_state: "expected",
            run_aiter=mock.Mock(return_value=(tensor, tensor)),
            compare_step=mock.Mock(),
        )
        candidate = SimpleNamespace(run=mock.Mock())
        with mock.patch("torch.cuda.synchronize"), \
             mock.patch("torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=0)):
            profile_dispatch.one_call(plugin, candidate, case, "aiter")
            profile_dispatch.one_call(plugin, candidate, case, "hip")
        self.assertEqual(plugin.run_aiter.call_count, 1)
        candidate.run.assert_called_once_with({"x": "gpu"}, tensor, tensor, 0)
        self.assertEqual(plugin.compare_step.call_count, 2)


if __name__ == "__main__":
    unittest.main()
