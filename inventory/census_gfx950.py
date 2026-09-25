#!/usr/bin/env python3
"""Index explicit gfx950 hints in a pinned AITER source checkout.

This is a search index, not a claim that each file is a dispatchable kernel.
Manual wrapper-to-kernel validation is required before task creation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".cu", ".cpp", ".cc", ".c", ".hip", ".h", ".hpp", ".cuh"}
SOURCE_ROOTS = ("aiter/", "csrc/")
GFX950 = re.compile(r"gfx950", re.IGNORECASE)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def census(repo: Path) -> dict:
    sha = git(repo, "rev-parse", "HEAD").decode().strip()
    if git(repo, "status", "--porcelain"):
        raise SystemExit("AITER checkout must be clean for a reproducible census")

    paths = [p.decode() for p in git(repo, "ls-files", "-z").split(b"\0") if p]
    rows = []
    for rel in sorted(paths):
        if not rel.startswith(SOURCE_ROOTS) or Path(rel).suffix not in SOURCE_SUFFIXES:
            continue
        path = repo / rel
        if not path.is_file():
            continue
        line_hits = [
            lineno
            for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
            if GFX950.search(line)
        ]
        if not line_hits and not GFX950.search(rel):
            continue
        rows.append(
            {
                "path": rel,
                "source_kind": "python" if path.suffix == ".py" else "native",
                "path_mentions_gfx950": bool(GFX950.search(rel)),
                "line_hits": line_hits,
            }
        )

    return {
        "schema_version": 1,
        "aiter_sha": sha,
        "scope": "git-tracked aiter/ and csrc/ source files with gfx950 in path or text",
        "interpretation": "source hints only; comments, guards and dead paths are not dispatch evidence",
        "counts": {
            "total": len(rows),
            "python": sum(row["source_kind"] == "python" for row in rows),
            "native": sum(row["source_kind"] == "native" for row in rows),
            "path_mentions_gfx950": sum(row["path_mentions_gfx950"] for row in rows),
        },
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aiter_checkout", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    args = parser.parse_args()
    result = census(args.aiter_checkout.resolve())
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
