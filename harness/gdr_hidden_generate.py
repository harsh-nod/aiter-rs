"""Generate a private GDR admission matrix on the trusted scoring host."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path


TASK_ID = "gfx950.gdr_decode_packed_bf16.stateful.v1"


def build_manifest() -> dict:
    rng = secrets.SystemRandom()
    repeated_slots = rng.sample(range(6), 4)
    mixed_slots = rng.sample(range(7), 3)
    mixed_indices = [mixed_slots[0], -1, 7, mixed_slots[1], 2**31 - 1, mixed_slots[2]]
    varied_slots = rng.sample(range(9), 4)
    varied_indices = varied_slots[:2] + [-2, 9] + varied_slots[2:] + [-1]
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "cases": [
            {
                "id": "hidden_01", "visibility": "withheld", "batch": 4, "pool": 6,
                "steps": 16, "seed": rng.randrange(1, 2**31), "indices": repeated_slots,
                "state_slot_padding": True,
            },
            {
                "id": "hidden_02", "visibility": "withheld", "batch": 6, "pool": 7,
                "steps": 1, "seed": rng.randrange(1, 2**31), "indices": mixed_indices,
                "strided_rows": True, "indices_stride": 2, "state_slot_padding": True,
            },
            {
                "id": "hidden_03", "visibility": "withheld", "batch": 1, "pool": 2,
                "steps": 1, "seed": rng.randrange(1, 2**31), "indices": [2**31 - 1],
            },
            {
                "id": "hidden_04", "visibility": "withheld", "batch": 7, "pool": 9,
                "steps": 3, "seed": rng.randrange(1, 2**31), "indices": varied_indices,
                "strided_rows": True, "state_slot_padding": True,
            },
        ],
        "benchmark_case_ids": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    args.output.chmod(0o600)
    print(json.dumps({"sha256": hashlib.sha256(payload.encode()).hexdigest(), "case_count": 4}))


if __name__ == "__main__":
    main()
