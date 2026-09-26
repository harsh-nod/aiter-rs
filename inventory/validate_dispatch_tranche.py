#!/usr/bin/env python3
"""Validate static source anchors and summarize a selected gfx950 tranche."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def validate(registry_path: Path, aiter_repo: Path) -> dict:
    registry = json.loads(registry_path.read_text())
    sha = subprocess.check_output(
        ["git", "-C", str(aiter_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != registry["aiter_sha"]:
        raise ValueError(f"AITER SHA mismatch: checkout={sha}, registry={registry['aiter_sha']}")
    if subprocess.check_output(
        ["git", "-C", str(aiter_repo), "status", "--porcelain"], text=True
    ):
        raise ValueError("AITER checkout must be clean")

    entry_ids: set[str] = set()
    regime_ids: set[str] = set()
    statuses: Counter[str] = Counter()
    regime_statuses: Counter[str] = Counter()
    for entry in registry["entries"]:
        entry_id = entry["id"]
        if entry_id in entry_ids:
            raise ValueError(f"duplicate entry ID: {entry_id}")
        entry_ids.add(entry_id)
        status = entry["trace_status"]
        if status not in {"source_traced", "partial"}:
            raise ValueError(f"unknown trace status for {entry_id}: {status}")
        statuses[status] += 1
        if not entry.get("dedupe") or not entry.get("evidence"):
            raise ValueError(f"missing dedupe rationale or source evidence: {entry_id}")

        for anchor in entry["evidence"]:
            rel = Path(anchor["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe evidence path: {rel}")
            path = aiter_repo / rel
            lines = path.read_text(errors="replace").splitlines()
            line = anchor["line"]
            if not isinstance(line, int) or not 1 <= line <= len(lines):
                raise ValueError(f"invalid line anchor: {rel}:{line}")
            if anchor["contains"] not in lines[line - 1]:
                raise ValueError(f"stale source anchor: {rel}:{line} lacks {anchor['contains']!r}")

        for regime in entry["regimes"]:
            regime_id = f"{entry_id}.{regime['id']}"
            if regime_id in regime_ids:
                raise ValueError(f"duplicate regime ID: {regime_id}")
            regime_ids.add(regime_id)
            if regime["status"] != "provisional" or not regime.get("difference"):
                raise ValueError(f"regime not provisional or missing distinction: {regime_id}")
            regime_statuses[status] += 1

    return {
        "aiter_sha": sha,
        "interpretation": "selected static source traces; no runtime dispatch, parity, or scored eligibility",
        "entry_count": len(entry_ids),
        "source_traced_entries": statuses["source_traced"],
        "partial_entries": statuses["partial"],
        "provisional_regimes": len(regime_ids),
        "source_traced_regimes": regime_statuses["source_traced"],
        "partial_regimes": regime_statuses["partial"],
        "scored_eligible": 0,
        "excluded_examples": len(registry["exclusions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aiter_checkout", type=Path)
    parser.add_argument(
        "--registry", type=Path, default=Path(__file__).with_name("gfx950_dispatch_tranche.json")
    )
    args = parser.parse_args()
    result = validate(args.registry.resolve(), args.aiter_checkout.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
