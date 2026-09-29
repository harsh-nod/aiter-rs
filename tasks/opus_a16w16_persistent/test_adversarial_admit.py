import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tasks.opus_a16w16_persistent import adversarial_admit as admit

HERE = Path(__file__).resolve().parent


class AdversarialAdmissionTests(unittest.TestCase):
    def test_only_public_matrix_case_is_selectable(self):
        matrix = json.loads((HERE / "adversarial_matrix_candidate.json").read_text())
        self.assertEqual(admit.select_public_case(matrix, "k-partial-final-tile")["k"], 194)
        with self.assertRaisesRegex(ValueError, "proposed public matrix"):
            admit.select_public_case(matrix, "not-a-public-case")

    def test_pinned_source_and_binary_hash_required_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter.so"
            adapter.write_bytes(b"wrong")
            args = SimpleNamespace(
                matrix=HERE / "adversarial_matrix_candidate.json",
                anchors=HERE / "cases.json",
                adapter_source=HERE / "source_adapter.hip",
                adapter=adapter,
                aiter_source=root,
            )
            with self.assertRaisesRegex(ValueError, "binary raw hash changed"):
                admit.check_provenance(args)

    def test_nonfinite_check_is_json_safe(self):
        value = admit._clean_check({"pass": False, "max_abs_error": float("nan")})
        self.assertIsNone(value["max_abs_error"])
        json.dumps(value, allow_nan=False)

    def test_adapter_status_is_not_silently_accepted(self):
        tensor = SimpleNamespace(data_ptr=lambda: 12)
        case = {"m": 256, "n": 256, "k": 128, "kid": 300}
        with self.assertRaisesRegex(RuntimeError, "hipError_t=1"):
            admit._adapter_call(lambda *args: 1, tensor, tensor, tensor, case, 0)

    def test_dispatch_requires_exact_persistent_specialization(self):
        name = "gemm_a16w16_persistent_kernel_traits_ELb1"
        self.assertTrue(admit._dispatch_matches([name], 300))
        self.assertFalse(admit._dispatch_matches([name], 1300))
        self.assertFalse(admit._dispatch_matches([name, "gemm_a16w16_other"], 300))

    def test_sku_pin_accepts_empty_torch_name_only_with_rocm_identity(self):
        host = "GPU[0]: Device Name: AMD Instinct MI350X\nGPU[0]: Device ID: 0x75a0\nGPU[0]: GUID: 39229"
        container = "GPU[0]: Device Name: N/A\nGPU[0]: Device ID: 0x75a0\nGPU[0]: GUID: 39229"
        self.assertEqual(admit._attest_sku("", "gfx950:sramecc+:xnack-", container, host), "AMD Instinct MI350X")
        with self.assertRaisesRegex(RuntimeError, "not jointly attested"):
            admit._attest_sku("", "gfx942", container, host)
        with self.assertRaisesRegex(RuntimeError, "not jointly attested"):
            admit._attest_sku("", "gfx950", container.replace("39229", "1"), host)


if __name__ == "__main__":
    unittest.main()
