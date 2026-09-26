"""Capture the host ROCm GPU identity before starting a scoring container."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def capture() -> dict:
    completed = subprocess.run(
        ["rocm-smi", "--showproductname"], capture_output=True, text=True, check=True, timeout=30
    )
    raw = completed.stdout
    names = re.findall(r"Card Series:\s*(.+)", raw)
    models = re.findall(r"Card Model:\s*(0x[0-9a-fA-F]+)", raw)
    arches = re.findall(r"GFX Version:\s*(gfx[0-9a-fA-F]+)", raw)
    if len(names) != 1 or len(models) != 1 or len(arches) != 1 or names[0].strip() == "N/A":
        raise RuntimeError("host ROCm report must identify exactly one named GPU")
    return {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu_name": names[0].strip(),
        "card_model": models[0].lower(),
        "arch": arches[0],
        "rocm_product_raw": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = capture()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    args.output.chmod(0o444)
    print(args.output)


if __name__ == "__main__":
    main()
