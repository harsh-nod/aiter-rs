#!/usr/bin/env python3
"""Count source-entry hints and gfx950 code objects at a pinned AITER SHA.

These are lexical/AST candidates, never a dispatch or semantic-type count.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


PINNED_SHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
DECORATORS = {"triton.jit", "gluon.jit", "flyc.kernel", "flyc.jit"}
NATIVE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".hip"}
GLOBAL_TOKEN = re.compile(r"\b__global__\b")
ARCH_PATH = re.compile(r"gfx\d+")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def arch_path_hint(path: str) -> str:
    architectures = set(ARCH_PATH.findall(path))
    if not architectures:
        return "generic_path_unverified_on_gfx950"
    if architectures == {"gfx950"}:
        return "explicit_gfx950_path"
    if "gfx950" in architectures:
        return "mixed_arch_path"
    return "explicit_other_arch_path"


def py_declarations(repo: Path, paths: list[str]) -> list[dict]:
    rows = []
    for rel in paths:
        if not rel.startswith("aiter/ops/") or not rel.endswith(".py"):
            continue
        tree = ast.parse((repo / rel).read_text(errors="replace"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                expr = decorator.func if isinstance(decorator, ast.Call) else decorator
                kind = ast.unparse(expr)
                if kind not in DECORATORS:
                    continue
                rows.append(
                    {
                        "path": rel,
                        "line": decorator.lineno,
                        "symbol": node.name,
                        "source_entry_hint": kind,
                        "arch_path_hint": arch_path_hint(rel),
                        "review_status": "unreviewed",
                    }
                )
    return rows


def native_tokens(repo: Path, paths: list[str]) -> list[dict]:
    rows = []
    for rel in paths:
        if not rel.startswith("csrc/") or Path(rel).suffix not in NATIVE_SUFFIXES:
            continue
        for line_no, line in enumerate((repo / rel).read_text(errors="replace").splitlines(), 1):
            for _ in GLOBAL_TOKEN.finditer(line):
                rows.append(
                    {
                        "path": rel,
                        "line": line_no,
                        "source_entry_hint": "cpp_global_token",
                        "arch_path_hint": arch_path_hint(rel),
                        "review_status": "unreviewed",
                    }
                )
    return rows


def census(repo: Path, python_index: Path) -> dict:
    sha = git(repo, "rev-parse", "HEAD").decode().strip()
    if sha != PINNED_SHA:
        raise SystemExit(f"expected AITER {PINNED_SHA}, found {sha}")
    if git(repo, "status", "--porcelain"):
        raise SystemExit("AITER checkout must be clean")
    py_index = json.loads(python_index.read_text())
    if py_index["aiter_sha"] != sha:
        raise SystemExit("Python index SHA differs from pinned AITER checkout")
    paths = sorted(p.decode() for p in git(repo, "ls-files", "-z").split(b"\0") if p)
    rows = py_declarations(repo, paths) + native_tokens(repo, paths)
    rows.sort(key=lambda row: (row["path"], row["line"], row["source_entry_hint"]))
    kinds = Counter(row["source_entry_hint"] for row in rows)
    arch_hints = Counter(row["arch_path_hint"] for row in rows)
    co_paths = [p for p in paths if p.startswith("hsa/gfx950/") and p.endswith(".co")]
    co_families = Counter(p.split("/")[2] for p in co_paths)
    all_co_paths = [p for p in paths if p.endswith(".co")]
    csv_paths = [p for p in paths if p.startswith("hsa/gfx950/") and p.endswith(".csv")]
    csv_rows = 0
    csv_families = Counter()
    csv_nonstandard_headers = []
    for rel in csv_paths:
        with (repo / rel).open(newline="") as handle:
            reader = csv.reader(
                line
                for line in handle
                if line.strip() and not line.lstrip().startswith(("#", "//"))
            )
            header = next(reader, [])
            if "co_name" not in header and "knl_name" not in header:
                csv_nonstandard_headers.append(rel)
            count = sum(1 for row in reader if row)
        csv_rows += count
        csv_families[rel.split("/")[2]] += count
    return {
        "schema_version": 1,
        "aiter_sha": sha,
        "interpretation": "source-entry hints and prebuilt artifacts, not verified gfx950 launches or semantic task types",
        "python_symbol_index": "inventory/gfx950_python_routes.json",
        "python_symbol_counts": {
            "indexed": py_index["coverage"]["indexed_symbols"],
            "reviewed": py_index["coverage"]["reviewed_symbols"],
            "unreviewed": py_index["coverage"]["unreviewed_symbols"],
        },
        "source_entry_hints": {
            "total": len(rows),
            "by_kind": dict(sorted(kinds.items())),
            "by_arch_path_hint": dict(sorted(arch_hints.items())),
            "caveat": "@triton.jit/@gluon.jit/@flyc.jit can be helpers; C++ __global__ is a lexical token and may be in comments, macros or non-gfx950 code; @flyc.kernel can still be unreachable",
        },
        "prebuilt_code_objects": {
            "all_architectures": len(all_co_paths),
            "gfx950": len(co_paths),
            "gfx950_by_family_directory": dict(sorted(co_families.items())),
            "caveat": "compiled files include tuning/layout variants and are not source-visible semantic task types",
        },
        "gfx950_csv_manifests": {
            "files": len(csv_paths),
            "data_rows": csv_rows,
            "data_rows_by_family_directory": dict(sorted(csv_families.items())),
            "nonstandard_headers": csv_nonstandard_headers,
            "caveat": "CSV rows are config/code-object selections, not independently reviewed algorithmic contracts",
        },
        "declaration_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aiter_checkout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    result = census(args.aiter_checkout.resolve(), here / "gfx950_python_routes.json")
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
