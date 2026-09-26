import json
import tempfile
import unittest
from pathlib import Path

from runs import remote_score
from runs.runner import canonical, digest


class RemoteScoreTests(unittest.TestCase):
    def test_gpu_sku_match_is_exact_for_scored_captures(self):
        matches = remote_score.frozen_gpu_sku_matches
        self.assertTrue(matches("AMD Instinct MI350X", "AMD Instinct MI350X", scored=True))
        self.assertTrue(matches("MI350X", "AMD Instinct MI350X", scored=False))
        self.assertFalse(matches("MI350X", "AMD Instinct MI350X", scored=True))
        self.assertFalse(matches("MI350", "AMD Instinct MI350X", scored=False))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        source = b'extern "C" int example() { return 0; }\n'
        sha = digest(source)
        (self.run / "blobs").mkdir()
        (self.run / "blobs" / sha).write_bytes(source)
        freeze = {
            "task_id": "quant-preview", "task_revision": "v1", "aiter_sha": "a" * 40,
            "gpu_sku": "MI350X", "target_arch": "gfx950", "harness_revision": "UNSET",
            "build_sources": ["kernel.hip"], "build_flags": ["-O3", "-shared", "-fPIC", "--offload-arch=gfx950"],
        }
        tree = digest(canonical({"kernel.hip": sha}))
        (self.run / "manifest.json").write_text(json.dumps({
            "run_id": "preview--v1--r001", "run_purpose": "unscored_preview",
            "incidence_eligible": False, "task_freeze": freeze,
            "task_freeze_sha256": digest(canonical(freeze)),
        }))
        (self.run / "result.json").write_text(json.dumps({
            "status": "completed", "incidence_eligible": False, "final_tree_sha256": tree,
        }))
        (self.run / "snapshots.jsonl").write_text(json.dumps({
            "sequence": 1, "tree_sha256": tree, "files": {"kernel.hip": sha},
        }) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_export_contains_only_verified_source_and_provenance(self):
        bundle_dir = self.root / "bundle"
        bundle = remote_score.export_bundle(self.run, bundle_dir)
        self.assertEqual(bundle["run_purpose"], "unscored_preview")
        self.assertFalse(bundle["incidence_eligible"])
        self.assertEqual((bundle_dir / "source" / "kernel.hip").read_bytes(), (self.run / "blobs" / bundle["source_files"]["kernel.hip"]).read_bytes())
        self.assertEqual(remote_score.validate_bundle(bundle_dir), bundle)
        self.assertEqual(sorted(path.name for path in bundle_dir.iterdir()), ["bundle.json", "source"])

    def test_export_rejects_tampered_agent_blob(self):
        blob = next((self.run / "blobs").iterdir())
        blob.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "blob hash mismatch"):
            remote_score.export_bundle(self.run, self.root / "bundle")

    def test_replay_rejects_tampered_export(self):
        bundle_dir = self.root / "bundle"
        remote_score.export_bundle(self.run, bundle_dir)
        (bundle_dir / "source" / "kernel.hip").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "files differ"):
            remote_score.validate_bundle(bundle_dir)

    def test_export_rejects_preview_labeled_scored(self):
        manifest = json.loads((self.run / "manifest.json").read_text())
        manifest["incidence_eligible"] = True
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incidence eligibility disagree"):
            remote_score.export_bundle(self.run, self.root / "bundle")

    def test_historical_snapshot_is_an_unscored_analyst_control(self):
        original = json.loads((self.run / "snapshots.jsonl").read_text())
        source = b'extern "C" int example() { return 1; }\n'
        sha = digest(source)
        (self.run / "blobs" / sha).write_bytes(source)
        final_tree = digest(canonical({"kernel.hip": sha}))
        final = {"sequence": 2, "tree_sha256": final_tree, "files": {"kernel.hip": sha}}
        (self.run / "snapshots.jsonl").write_text(json.dumps(original) + "\n" + json.dumps(final) + "\n")
        result = json.loads((self.run / "result.json").read_text())
        result["final_tree_sha256"] = final_tree
        (self.run / "result.json").write_text(json.dumps(result))

        bundle_dir = self.root / "bundle"
        bundle = remote_score.export_bundle(self.run, bundle_dir, snapshot_sequence=1)
        self.assertEqual(bundle["run_purpose"], "unscored_preview")
        self.assertFalse(bundle["incidence_eligible"])
        self.assertTrue(bundle["analyst_control"])
        self.assertEqual(bundle["original_final_tree_sha256"], final_tree)
        self.assertEqual(bundle["final_tree_sha256"], original["tree_sha256"])
        self.assertEqual(remote_score.validate_bundle(bundle_dir), bundle)
        with self.assertRaisesRegex(ValueError, "uniquely precede"):
            remote_score.export_bundle(self.run, self.root / "other", snapshot_sequence=2)


if __name__ == "__main__":
    unittest.main()
