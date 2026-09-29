import tempfile
import unittest
import subprocess
from pathlib import Path

from prepare_overlay import candidate_bytes, changed_paths


class OverlayInputTest(unittest.TestCase):
    def test_regular_header_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.cuh"
            path.write_text("#pragma once\n")
            self.assertEqual(candidate_bytes(path), b"#pragma once\n")

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cuh"
            source.write_text("#pragma once\n")
            link = root / "link.cuh"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):
                candidate_bytes(link)

    def test_nul_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.cuh"
            path.write_bytes(b"x\x00y")
            with self.assertRaisesRegex(ValueError, "NUL"):
                candidate_bytes(path)

    def test_git_detects_extra_changed_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            header = root / "allowed.cuh"
            header.write_text("old\n")
            subprocess.run(["git", "-C", str(root), "add", "allowed.cuh"], check=True)
            subprocess.run([
                "git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "base",
            ], check=True)
            header.write_text("new\n")
            self.assertEqual(changed_paths(root), ["allowed.cuh"])
            (root / "unexpected.cuh").write_text("bad\n")
            self.assertEqual(set(changed_paths(root)), {"allowed.cuh", "unexpected.cuh"})


if __name__ == "__main__":
    unittest.main()
