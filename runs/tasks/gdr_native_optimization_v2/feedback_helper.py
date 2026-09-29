#!/usr/bin/env python3
"""Agent-visible client for bounded public-only GDR feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


REQUEST_SCHEMA = "aiter-rs-gdr-opt-public-request-v1"
RESPONSE_SCHEMA = "aiter-rs-gdr-opt-public-response-v1"
MAX_REQUESTS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("correctness", "benchmark"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--wait-seconds", type=int, default=360)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 600:
        parser.error("wait-seconds must be 1-600")
    workspace = args.workspace.resolve()
    requests = workspace / ".feedback/requests"
    responses = workspace / ".feedback/responses"
    if not requests.is_dir() or not responses.is_dir() or requests.is_symlink() or responses.is_symlink():
        parser.error("trusted public feedback broker is not active")
    issued = [int(path.stem) for path in requests.glob("[0-9][0-9][0-9][0-9].json")]
    request_id = max(issued, default=0) + 1
    if request_id > MAX_REQUESTS:
        parser.error("public feedback request limit reached")
    source_sha = hashlib.sha256((workspace / "kernel.hip").read_bytes()).hexdigest()
    payload = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "source_sha256": source_sha,
        "kind": args.kind,
    }
    temporary = requests / f".{request_id:04d}.{os.getpid()}.tmp"
    final = requests / f"{request_id:04d}.json"
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    try:
        os.link(temporary, final)
    finally:
        temporary.unlink()
    response_path = responses / f"{request_id:04d}.json"
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        if response_path.is_file() and not response_path.is_symlink():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if response.get("schema") != RESPONSE_SCHEMA or response.get("request_id") != request_id:
                parser.error("public feedback response identity mismatch")
            if response.get("source_sha256") not in (source_sha, None):
                parser.error("public feedback source hash mismatch")
            print(json.dumps(response, sort_keys=True))
            return 0 if response.get("status") == "complete" else 1
        time.sleep(0.05)
    parser.error("public feedback timed out")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
