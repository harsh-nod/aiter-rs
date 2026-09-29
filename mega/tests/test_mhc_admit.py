import unittest

import torch

from mega.mhc_admit import compare_output, expected_dispatch, record_dispatch


class MhcAdmissionTests(unittest.TestCase):
    def test_expected_dispatch_distinguishes_norm_and_fallback(self):
        self.assertEqual(
            expected_dispatch({"route": "fused"}),
            {"mhc_fused_post_pre_gemm_sqrsum", "mhc_pre_big_fuse"},
        )
        self.assertEqual(
            expected_dispatch({"route": "fused", "norm": True}),
            {"mhc_fused_post_pre_gemm_sqrsum", "mhc_pre_big_fuse_rmsnorm"},
        )
        self.assertEqual(
            expected_dispatch({"route": "large_m_fallback"}),
            {"mhc_fused_post_pre_large_m", "mhc_post", "mhc_pre", "mhc_pre_gemm_sqrsum", "mhc_pre_big_fuse"},
        )

    def test_compare_output_requires_all_elements_and_matching_dtype(self):
        expected = torch.ones(20, dtype=torch.float32)
        actual = expected.clone()
        actual[-1] = 1.1
        result = compare_output(expected, actual, "post_mix")
        self.assertFalse(result["pass"])
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(compare_output(expected, expected.to(torch.bfloat16), "post_mix")["reason"], "shape_or_dtype")

    def test_record_dispatch_wraps_all_entrypoints(self):
        class Module:
            pass

        module = Module()
        for name in (
            "mhc_fused_post_pre_gemm_sqrsum", "mhc_pre_gemm_sqrsum", "mhc_pre_big_fuse",
            "mhc_pre_big_fuse_rmsnorm", "mhc_fused_post_pre_large_m",
            "mhc_post", "mhc_pre",
        ):
            setattr(module, name, lambda: None)
        with record_dispatch(module) as calls:
            module.mhc_fused_post_pre_gemm_sqrsum()
            module.mhc_pre_big_fuse()
        self.assertEqual(calls, ["mhc_fused_post_pre_gemm_sqrsum", "mhc_pre_big_fuse"])


if __name__ == "__main__":
    unittest.main()
