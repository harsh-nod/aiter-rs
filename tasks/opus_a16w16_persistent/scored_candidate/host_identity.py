"""Trusted minimal NSS files for non-root OPUS Docker workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def write_nss(root: Path, uid: int, gid: int) -> tuple[Path, Path]:
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077:
        raise ValueError("NSS directory must be a mode-700 private directory")
    contents = {
        "container-passwd": (
            "root:x:0:0:root:/root:/bin/sh\n"
            "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
            f"aiter-replay:x:{uid}:{gid}:aiter replay:/tmp:/bin/sh\n"
        ),
        "container-group": (
            "root:x:0:\n"
            "nogroup:x:65534:\n"
            f"aiter-replay:x:{gid}:\n"
        ),
    }
    for name, value in contents.items():
        path = root / name
        if path.is_symlink():
            raise ValueError("NSS file cannot be a symlink")
        if path.exists():
            if path.read_text(encoding="ascii") != value:
                raise ValueError("NSS file differs from trusted host identity")
            continue
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(value)
    return root / "container-passwd", root / "container-group"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    write_nss(args.private_root.resolve(), os.getuid(), os.getgid())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
