import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tasks.gdr_native_optimization import parse_dispatch_traces as parser


class DispatchParserTests(unittest.TestCase):
    def make_trace(self, root: Path, *, grid: int = 131072, function: str = "hipLaunchKernel") -> None:
        tag = "valid_aiter_04"
        receipt = {
            "mode": "aiter", "case_id": "valid_slots", "correctness_and_guards": "pass",
            "batch": 4, "expected_grid_blocks": 512, "expected_workgroup_threads": 256,
        }
        (root / f"receipt_{tag}.json").write_text(json.dumps(receipt), encoding="utf-8")
        profile = root / f"profile_{tag}" / "subprocess"
        profile.mkdir(parents=True)
        with (profile / "one_kernel_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "Kernel_Name", "Correlation_Id", "Thread_Id", "Workgroup_Size_X",
                "Workgroup_Size_Y", "Workgroup_Size_Z", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
            ))
            writer.writeheader()
            writer.writerow({
                "Kernel_Name": "(anonymous namespace)::gdr_decode_packed_bf16_kernel(...)",
                "Correlation_Id": 5, "Thread_Id": 7, "Workgroup_Size_X": 256,
                "Workgroup_Size_Y": 1, "Workgroup_Size_Z": 1,
                "Grid_Size_X": grid, "Grid_Size_Y": 1, "Grid_Size_Z": 1,
            })
        with (profile / "one_hip_api_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("Correlation_Id", "Process_Id", "Function"))
            writer.writeheader()
            writer.writerow({"Correlation_Id": 5, "Process_Id": 7, "Function": function})

    def test_structured_trace_checks_kernel_count_geometry_and_hip_correlation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_trace(root)
            entry = parser.parse_one(root, "valid_aiter_04", "aiter", "valid_slots")
            self.assertEqual(entry["grid_blocks"], 512)
            self.assertEqual(entry["grid_work_items"], 131072)
            self.assertEqual(entry["launch_api"], "hipLaunchKernel")
            self.assertEqual(entry["target_dispatch_count"], 1)

    def test_wrong_geometry_or_api_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_trace(root, grid=512)
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                parser.parse_one(root, "valid_aiter_04", "aiter", "valid_slots")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_trace(root, function="hipMemcpy")
            with self.assertRaisesRegex(ValueError, "hipLaunchKernel correlation"):
                parser.parse_one(root, "valid_aiter_04", "aiter", "valid_slots")

    def test_all_modes_must_name_same_kernel(self):
        entries = [
            {"kernel_name": "gdr"}, {"kernel_name": "different"},
        ]
        with mock.patch.object(parser, "JOBS", (("a", "aiter", "x"), ("b", "hip", "x"))), \
             mock.patch.object(parser, "parse_one", side_effect=entries):
            with self.assertRaisesRegex(ValueError, "kernel identity differs"):
                parser.parse_all(Path("/unused"))


if __name__ == "__main__":
    unittest.main()
