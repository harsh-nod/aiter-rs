"""CPU-only tests for the sparse-prefill candidate scorer guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from references.sparse_prefill_scorer import (
    HipCandidate,
    PRIVATE_MATRIX_SHA256,
    POISON,
    compare_output,
    require_outside_repo,
)
from references.sparse_prefill_gfx950 import make_inputs, public_cases


def _compare(actual: torch.Tensor, expected: torch.Tensor, **changes) -> dict:
    options = {
        "guard_before": torch.full((8,), POISON, dtype=torch.bfloat16),
        "guard_after": torch.full((8,), POISON, dtype=torch.bfloat16),
        "inputs_unchanged": True,
        "return_code": 0,
    }
    options.update(changes)
    return compare_output(actual, expected, **options)


def test_exact_output_passes() -> None:
    expected = torch.ones((2, 16, 512), dtype=torch.bfloat16)
    result = _compare(expected.clone(), expected)
    assert result["passed"]
    assert result["mismatch_fraction"] == 0.0


@pytest.mark.parametrize("fault", ["wrong", "nonfinite", "no_write", "guard", "input", "return"])
def test_guarded_comparison_rejects_fault(fault: str) -> None:
    expected = torch.ones((2, 16, 512), dtype=torch.bfloat16)
    actual = expected.clone()
    options = {}
    if fault == "wrong":
        actual.zero_()
    elif fault == "nonfinite":
        actual[0, 0, 0] = float("nan")
    elif fault == "no_write":
        actual.fill_(POISON)
    elif fault == "guard":
        before = torch.full((8,), POISON, dtype=torch.bfloat16)
        before[0] = 0
        options["guard_before"] = before
    elif fault == "input":
        options["inputs_unchanged"] = False
    else:
        options["return_code"] = 7
    assert not _compare(actual, expected, **options)["passed"]


def test_sink_only_zero_passes() -> None:
    expected = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
    assert _compare(expected.clone(), expected)["passed"]


def test_private_commitment_matches_receipt() -> None:
    receipt = json.loads(
        (Path(__file__).resolve().parents[1] / "tasks/sparse_prefill_gfx950/admission-receipt.json").read_text()
    )
    assert PRIVATE_MATRIX_SHA256 == receipt["withheld_probe"]["matrix_sha256_commitment"]


def test_repo_output_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        require_outside_repo(Path(__file__).resolve().parents[1] / "private-output")
    require_outside_repo(tmp_path / "private-output")


def test_candidate_abi_argument_order_without_gpu(monkeypatch) -> None:
    case = public_cases()[0]
    inputs = make_inputs(case)
    output = torch.empty_like(inputs["q"])
    captured = []
    candidate = object.__new__(HipCandidate)
    candidate.function = lambda *args: captured.extend(args) or 0
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: type("Stream", (), {"cuda_stream": 123})()
    )
    assert candidate.run(inputs, output, case.softmax_scale) == 0
    assert len(captured) == 17
    keys = (
        "q", "unified_kv", "kv_indices_prefix", "kv_indptr_prefix", "kv",
        "kv_indices_extend", "kv_indptr_extend", "attn_sink",
    )
    assert [arg.value for arg in captured[:8]] == [inputs[key].data_ptr() for key in keys]
    assert captured[8].value == output.data_ptr()
    assert tuple(captured[9:15]) == (
        case.n, case.h, case.pages, case.extend_rows,
        inputs["kv_indices_prefix"].numel(), inputs["kv_indices_extend"].numel(),
    )
    assert captured[15] == case.softmax_scale
    assert captured[16].value == 123
