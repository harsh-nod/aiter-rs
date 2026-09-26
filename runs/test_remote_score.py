import json
import tempfile
import unittest
from pathlib import Path

from runs import remote_score
from runs.runner import canonical, digest


class RemoteScoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
