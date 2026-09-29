import contextlib
import hashlib
import types
import unittest
from pathlib import Path
from unittest import mock

from references.quant_mxfp4_graph_control import (
    GRAPH_CALLS,
    GuardedInput,
    SPEC_RAW_SHA256,
    capture_graph,
    exact_bytes,
    paired_samples,
    quiet_gpu,
    summarize_samples,
)


class QuantGraphControlTests(unittest.TestCase):
    def test_pinned_public_spec(self):
        path = Path(__file__).with_name("quant_mxfp4_gfx950.json")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), SPEC_RAW_SHA256)

    def test_guarded_input_catches_mutation_and_both_boundaries(self):
        import torch

        guarded = GuardedInput(torch.arange(32, dtype=torch.float16), device="cpu")
        self.assertTrue(all(guarded.check().values()))
        guarded.tensor[0] = -1
        self.assertFalse(guarded.check()["input_unchanged"])
        guarded.raw[guarded.start - 1] = 0
        guarded.raw[guarded.end] = 0
        checks = guarded.check()
        self.assertFalse(checks["prefix_guard"])
        self.assertFalse(checks["suffix_guard"])

    def test_exact_bytes_reports_offsets_and_length(self):
        result = exact_bytes(b"\x00\x02\x03", b"\x00\x01\x03")
        self.assertFalse(result["pass"])
        self.assertEqual(result["first_mismatch_offsets"], [1])
        self.assertFalse(exact_bytes(b"\x00", b"\x00\x01")["pass"])

    def test_capture_records_exactly_32_calls(self):
        class FakeCuda:
            def CUDAGraph(self):
                return object()

            @contextlib.contextmanager
            def graph(self, unused):
                yield

        count = 0

        def op():
            nonlocal count
            count += 1

        capture_graph(types.SimpleNamespace(cuda=FakeCuda()), op)
        self.assertEqual(count, GRAPH_CALLS)

    def test_paired_samples_alternate_and_amortize(self):
        calls = []

        def baseline():
            calls.append("a")

        def candidate():
            calls.append("c")

        def timed(fn):
            fn()
            return 32.0 if fn is baseline else 64.0

        samples = paired_samples(timed, baseline, candidate, repeats=4)
        self.assertEqual(calls, ["a", "c", "c", "a", "a", "c", "c", "a"])
        self.assertEqual(samples["aiter_ms"], [1.0] * 4)
        self.assertEqual(samples["candidate_ms"], [2.0] * 4)

    def test_noise_and_ratio_gates(self):
        samples = {"aiter_ms": [1.0, 1.01, 0.99], "candidate_ms": [1.03, 1.04, 1.02]}
        result = summarize_samples(samples)
        self.assertTrue(result["noise_qualified"])
        self.assertTrue(result["exploratory_threshold_pass"])
        samples["candidate_ms"] = [1.1, 1.11, 1.09]
        self.assertFalse(summarize_samples(samples)["exploratory_threshold_pass"])
        samples["aiter_ms"] = [1.0, 1.2, 1.4]
        self.assertFalse(summarize_samples(samples)["noise_qualified"])

    def test_quiet_gate_requires_host_pid_mapping_and_no_foreign_process(self):
        path = "references.quant_mxfp4_graph_control"
        with mock.patch(path + "._command", return_value="KFD process information"), \
             mock.patch(path + ".os.getpid", return_value=42), \
             mock.patch(path + "._active_gpu_pids", return_value=[42]):
            self.assertTrue(quiet_gpu()["self_pid_visible"])
        with mock.patch(path + "._command", return_value="KFD process information"), \
             mock.patch(path + ".os.getpid", return_value=42), \
             mock.patch(path + "._active_gpu_pids", return_value=[99]):
            with self.assertRaisesRegex(RuntimeError, "host PID not visible"):
                quiet_gpu()
        with mock.patch(path + "._command", return_value="KFD process information"), \
             mock.patch(path + ".os.getpid", return_value=42), \
             mock.patch(path + "._active_gpu_pids", return_value=[42, 99]):
            with self.assertRaisesRegex(RuntimeError, "foreign active"):
                quiet_gpu()


if __name__ == "__main__":
    unittest.main()
