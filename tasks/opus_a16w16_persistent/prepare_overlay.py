"""Trusted, single-header OPUS candidate overlay into a new pinned AITER tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
from pathlib import Path

from admit import checked_aiter_sha

AITERSHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
HEADER = Path("csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh")
BASE_HEADER_SHA256 = "bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b"
MAX_HEADER_BYTES = 256 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_bytes(path: Path) -> bytes:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("candidate header must be a regular file, not a symlink or directory")
    if path.stat().st_size > MAX_HEADER_BYTES:
        raise ValueError("candidate header exceeds size cap")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("candidate header contains NUL bytes")
    return data


def changed_paths(tree: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    return [line[3:] for line in output.splitlines()]


def prepare(source: Path, candidate: Path, output: Path) -> dict:
    source, candidate, output = source.resolve(), candidate.absolute(), output.resolve()
    script_dir = Path(__file__).resolve().parent
    task_repo = (script_dir.parents[1]
                 if script_dir.name == "opus_a16w16_persistent" and script_dir.parent.name == "tasks"
                 else script_dir)
    if (output.exists() or output.is_relative_to(source) or
        output.is_relative_to(candidate.parent.resolve()) or output.is_relative_to(task_repo)):
        raise ValueError("overlay output must be a new directory outside inputs")
    checked_aiter_sha(source, AITERSHA)
    original = (source / HEADER).read_bytes()
    if digest(original) != BASE_HEADER_SHA256:
        raise ValueError("pinned header hash mismatch")
    new_bytes = candidate_bytes(candidate)
    shutil.copytree(source, output, symlinks=True)
    target = output / HEADER
    if target.is_symlink() or not target.is_file():
        raise ValueError("overlay target is not a regular file")
    target.write_bytes(new_bytes)
    paths = changed_paths(output)
    if paths not in ([], [HEADER.as_posix()]):
        raise ValueError(f"overlay modified unexpected paths: {paths}")
    return {
        "schema": "aiter-rs-opus-persistent-single-header-overlay-v1",
        "aiter_sha": AITERSHA,
        "allowed_path": HEADER.as_posix(),
        "base_header_sha256": BASE_HEADER_SHA256,
        "candidate_header_sha256": digest(new_bytes),
        "changed_paths": paths,
        "output_tree": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-source", type=Path, required=True)
    parser.add_argument("--candidate-header", type=Path, required=True)
    parser.add_argument("--output-tree", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.aiter_source, args.candidate_header, args.output_tree), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
