#!/usr/bin/env python3
"""Index Python-exposed AITER symbols without treating them as kernel types."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


PINNED_SHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
GFX = re.compile(r"gfx950", re.IGNORECASE)
ARCH_DISPATCH = re.compile(r"get_gfx_runtime|gcnArchName|is_gfx950|gfx950", re.I)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def validate_anchor(repo: Path, anchor: dict) -> None:
    lines = (repo / anchor["path"]).read_text(errors="replace").splitlines()
    line = anchor["line"]
    if not 1 <= line <= len(lines) or anchor["contains"] not in lines[line - 1]:
        raise ValueError(f"stale source anchor: {anchor}")


def symbol_row(path: str, node: ast.AST, lines: list[str]) -> dict:
    name = node.name
    native_binding = None
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or ast.unparse(decorator.func) != "compile_ops":
            continue
        module = (
            decorator.args[0].value
            if decorator.args and isinstance(decorator.args[0], ast.Constant)
            else None
        )
        export = next(
            (
                kw.value.value
                for kw in decorator.keywords
                if kw.arg == "fc_name" and isinstance(kw.value, ast.Constant)
            ),
            name,
        )
        native_binding = {"module": module, "export": export}
        break
    native_stub = native_binding is not None
    body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    call_names = sorted(
        {
            ast.unparse(n.func)
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and any(
                token in ast.unparse(n.func).lower()
                for token in ("hip", "opus", "triton", "flydsl", "torch.ops")
            )
        }
    )
    parts = Path(path).with_suffix("").parts
    backend_hints = []
    if native_stub:
        backend_hints.append("compile_ops_native_binding")
    if "flydsl" in parts:
        backend_hints.append("flydsl_path")
    if "triton" in parts:
        backend_hints.append("triton_path")
    if call_names:
        backend_hints.append("backend_named_call")
    if isinstance(node, ast.ClassDef):
        methods = [
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if "forward" in methods or "__call__" in methods:
            backend_hints.append("callable_class")
    row = {
        "key": f"{path}::{name}",
        "path": path,
        "symbol": name,
        "line": node.lineno,
        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
        "visibility": "public" if not name.startswith("_") else "private_binding_stub",
        "entry_scope": "ops_facade_module" if len(parts) == 3 else "nested_ops_module",
        "backend_hints": backend_hints,
        "backend_named_calls": call_names,
        "gfx950_text_hint": bool(GFX.search(body)),
        "arch_dispatch_text_hint": bool(ARCH_DISPATCH.search(body)),
        "review_status": "unreviewed",
        "trace_confidence": "none",
        "source_visibility": "unknown",
    }

    if native_binding is not None:
        row["native_binding"] = native_binding
    return row


def census(repo: Path, reviews_path: Path, tranche_path: Path) -> dict:
    sha = git(repo, "rev-parse", "HEAD").decode().strip()
    if sha != PINNED_SHA:
        raise SystemExit(f"expected AITER {PINNED_SHA}, found {sha}")
    if git(repo, "status", "--porcelain"):
        raise SystemExit("AITER checkout must be clean")
    reviews = json.loads(reviews_path.read_text())
    tranche = json.loads(tranche_path.read_text())
    tranche_ids = {entry["id"] for entry in tranche["entries"]}
    if reviews["aiter_sha"] != sha or tranche["aiter_sha"] != sha:
        raise ValueError("review or tranche SHA differs from AITER checkout")
    by_key = {}
    for review in reviews["reviews"]:
        key = review["key"]
        if key in by_key:
            raise ValueError(f"duplicate review key: {key}")
        if "tranche_id" in review and review["tranche_id"] not in tranche_ids:
            raise ValueError(f"unknown tranche ID: {review['tranche_id']}")
        for anchor in review.get("evidence", []):
            validate_anchor(repo, anchor)
        by_key[key] = review

    paths = [p.decode() for p in git(repo, "ls-files", "-z").split(b"\0") if p]
    py_paths = sorted(p for p in paths if p.startswith("aiter/ops/") and p.endswith(".py"))
    entries = []
    skipped_private = 0
    parse_errors = []
    for rel in py_paths:
        lines = (repo / rel).read_text(errors="replace").splitlines()
        try:
            tree = ast.parse("\n".join(lines) + "\n", filename=rel)
        except SyntaxError as exc:
            parse_errors.append({"path": rel, "line": exc.lineno, "message": exc.msg})
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            native_stub = any("compile_ops" in ast.unparse(d) for d in node.decorator_list)
            if node.name.startswith("_") and not native_stub:
                skipped_private += 1
                continue
            row = symbol_row(rel, node, lines)
            review = by_key.pop(row["key"], None)
            if review is not None:
                row["review_status"] = "reviewed"
                row["trace_confidence"] = review["trace_confidence"]
                row["source_visibility"] = review["source_visibility"]
                row["review_id"] = review["id"]
                row["architecture_scope"] = review["architecture_scope"]
                row["backend_routes"] = review["backend_routes"]
                if "tranche_id" in review:
                    row["tranche_id"] = review["tranche_id"]
            entries.append(row)
    if by_key:
        raise ValueError(f"reviewed symbols absent from AST index: {sorted(by_key)}")
    entries.sort(key=lambda row: row["key"])
    status_counts = Counter(row["review_status"] for row in entries)
    backend_hint_counts = Counter(h for row in entries for h in row["backend_hints"])
    return {
        "schema_version": 1,
        "aiter_sha": sha,
        "scope": "git-tracked aiter/ops/**/*.py module-level public symbols and private compile_ops stubs",
        "warning": "Discovery index, not a dispatchable-kernel count or task-type denominator; every unreviewed route has unknown backend and source visibility",
        "coverage": {
            "python_modules_scanned": len(py_paths),
            "parse_errors": parse_errors,
            "excluded_private_nondecorated_symbols": skipped_private,
            "indexed_symbols": len(entries),
            "public_symbols": sum(row["visibility"] == "public" for row in entries),
            "ops_facade_module_symbols": sum(row["entry_scope"] == "ops_facade_module" for row in entries),
            "nested_ops_module_symbols": sum(row["entry_scope"] == "nested_ops_module" for row in entries),
            "private_compile_ops_stubs": sum(row["visibility"] == "private_binding_stub" for row in entries),
            "reviewed_symbols": status_counts["reviewed"],
            "unreviewed_symbols": status_counts["unreviewed"],
            "backend_hint_counts": dict(sorted(backend_hint_counts.items())),
            "gfx950_text_hint_symbols": sum(row["gfx950_text_hint"] for row in entries),
        },
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aiter_checkout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    result = census(
        args.aiter_checkout.resolve(),
        here / "python_route_reviews.json",
        here / "gfx950_dispatch_tranche.json",
    )
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
