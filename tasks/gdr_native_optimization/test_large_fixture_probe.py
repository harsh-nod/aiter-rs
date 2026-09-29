import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness.core import read_spec
from references import gdr_decode_packed_bf16 as plugin
from tasks.gdr_native_optimization import large_fixture_probe as probe


SPEC = Path(plugin.__file__).with_name("gdr_decode_packed_bf16_gfx950.json")
FIXTURE = Path(__file__).with_name("large_valid_slots_candidate.json")


class LargeFixtureProbeTests(unittest.TestCase):
    def test_fixture_is_pinned_public_and_outside_original_matrix(self):
        spec = read_spec(SPEC)
        case = probe.load_fixture(FIXTURE, spec)
        self.assertEqual(case["batch"], 16)
        self.assertEqual(case["pool"], 20)
        self.assertEqual(len(set(case["indices"])), 16)
        self.assertNotIn(case["id"], {entry["id"] for entry in spec["cases"]})
        self.assertEqual(len(spec["cases"]), 4)

    def test_fixture_cpu_oracle_mutates_only_selected_slots(self):
        import torch

        case = probe.load_fixture(FIXTURE, read_spec(SPEC))
        state = plugin.make_initial_state(case)
        before = state.clone()
        inputs = plugin.make_step_inputs(case, 0)
        output = plugin.oracle_step(inputs, case["indices"], state)
        self.assertEqual(tuple(output.shape), (16, 1, 32, 128))
        self.assertTrue(bool(torch.isfinite(output.float()).all()))
        selected = set(case["indices"])
        for slot in range(case["pool"]):
            if slot not in selected:
                self.assertTrue(torch.equal(state[slot], before[slot]))
        self.assertTrue(any(not torch.equal(state[slot], before[slot]) for slot in selected))

    def test_quiet_gpu_requires_host_pid_after_runtime_init(self):
        with mock.patch.object(probe, "_command", return_value="KFD process information"), \
             mock.patch.object(probe, "_active_gpu_pids", return_value=[42]), \
             mock.patch.object(probe.os, "getpid", return_value=42):
            self.assertTrue(probe.quiet_gpu(require_self=True)["self_visible"])
        with mock.patch.object(probe, "_command", return_value="KFD process information"), \
             mock.patch.object(probe, "_active_gpu_pids", return_value=[]), \
             mock.patch.object(probe.os, "getpid", return_value=42):
            with self.assertRaisesRegex(RuntimeError, "host PID not visible"):
                probe.quiet_gpu(require_self=True)

    def test_improvement_is_distinct_from_non_inferiority(self):
        base = {name: {"ratio": 1.0} for name in probe.BUCKETS}
        performance = {"pass": True, "buckets": base}
        probe.summarize_improvement(performance)
        self.assertFalse(performance["proposed_5pct_improvement_pass"])
        faster = {name: {"ratio": 0.94} for name in probe.BUCKETS}
        performance = {"pass": True, "buckets": faster}
        probe.summarize_improvement(performance)
        self.assertTrue(performance["proposed_5pct_improvement_pass"])

    def test_probe_gates_all_five_oracle_cases_before_three_graph_buckets(self):
        spec = read_spec(SPEC)
        fixture = probe.load_fixture(FIXTURE, spec)
        args = SimpleNamespace(
            spec=SPEC, fixture=FIXTURE, source=Path("/unused/source.hip"),
            candidate=Path("/unused/candidate.so"), aiter_source=Path("/unused/aiter"),
            host_gpu_report=Path("/unused/host.json"),
        )
        fake_plugin = SimpleNamespace(load_hip_candidate=lambda path: object())
        seen_cases = []
        timed_cases = []
        fail_fixture = False

        def case_result(unused_plugin, unused_candidate, case):
            seen_cases.append(case["id"])
            return {"id": case["id"], "pass": not (fail_fixture and case["id"] == fixture["id"])}

        def timed_result(unused_plugin, unused_candidate, case, calls):
            timed_cases.append((case["id"], calls))
            return {"aiter_ms": [1.0, 1.0, 1.0], "candidate_ms": [1.0, 1.0, 1.0]}

        hashes = {
            SPEC: probe.SPEC_SHA256,
            args.source: probe.SOURCE_SHA256,
            args.candidate: probe.BINARY_SHA256,
        }
        with mock.patch.object(probe, "read_spec", return_value=spec), \
             mock.patch.object(probe, "raw_sha256", side_effect=lambda path: hashes[path]), \
             mock.patch.object(probe, "load_fixture", return_value=fixture), \
             mock.patch.object(probe, "_command", return_value=probe.AITERSHA), \
             mock.patch.object(probe, "quiet_gpu", return_value={"self_visible": True, "foreign_active_count": 0}), \
             mock.patch.object(probe.importlib, "import_module", return_value=fake_plugin), \
             mock.patch.object(probe, "_gpu_manifest", return_value={}), \
             mock.patch.object(probe, "run_case", side_effect=case_result), \
             mock.patch.object(probe, "benchmark_case", side_effect=timed_result), \
             mock.patch.object(sys, "path", list(sys.path)):
            result = probe.run_probe(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(seen_cases), 5)
            self.assertEqual(timed_cases, [(name, 32) for name in probe.BUCKETS])
            self.assertTrue(result["post_graph_oracle_and_guards_pass"])
            self.assertFalse(result["performance"]["proposed_5pct_improvement_pass"])
            fail_fixture = True
            timed_cases.clear()
            failed = probe.run_probe(args)
        self.assertEqual(failed["status"], "correctness_failed")
        self.assertEqual(failed["performance"]["status"], "not_run")
        self.assertEqual(timed_cases, [])


if __name__ == "__main__":
    unittest.main()
