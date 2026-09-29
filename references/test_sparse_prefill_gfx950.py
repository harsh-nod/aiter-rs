"""CPU-only tests for the independent sparse-prefill admission oracle."""

from __future__ import annotations

import pytest
import torch

from references.sparse_prefill_gfx950 import D, Case, make_inputs, oracle, public_cases, validate_inputs


def test_sink_is_denominator_only() -> None:
    inputs = {
        "q": torch.zeros((1, 16, D), dtype=torch.bfloat16),
        "unified_kv": torch.ones((1, D), dtype=torch.bfloat16),
        "kv_indices_prefix": torch.tensor([0], dtype=torch.int32),
        "kv_indptr_prefix": torch.tensor([0, 1], dtype=torch.int32),
        "kv": torch.full((1, D), 3, dtype=torch.bfloat16),
        "kv_indices_extend": torch.tensor([0], dtype=torch.int32),
        "kv_indptr_extend": torch.tensor([0, 1], dtype=torch.int32),
        "attn_sink": torch.zeros((16,), dtype=torch.float32),
    }
    actual = oracle(inputs, 0.0)
    assert torch.equal(actual, torch.full_like(actual, 4 / 3))
    inputs["kv_indices_prefix"] = torch.tensor([0, 0], dtype=torch.int32)
    inputs["kv_indptr_prefix"] = torch.tensor([0, 2], dtype=torch.int32)
    expected = torch.full_like(actual, 1.25)
    assert torch.equal(oracle(inputs, 0.0), expected)


def test_sink_only_is_zero() -> None:
    case = Case("empty", 1, 2, 16, 4, 4, "empty")
    inputs = make_inputs(case)
    assert torch.equal(oracle(inputs, case.softmax_scale), torch.zeros((2, 16, D), dtype=torch.bfloat16))


@pytest.mark.parametrize("case", public_cases(), ids=lambda case: case.name)
def test_public_case_is_deterministic_and_finite(case: Case) -> None:
    first, second = make_inputs(case), make_inputs(case)
    assert all(torch.equal(first[key], second[key]) for key in first)
    result = oracle(first, case.softmax_scale)
    assert result.shape == (case.n, case.h, D)
    assert result.dtype == torch.bfloat16
    assert bool(torch.isfinite(result).all())
    if case.pattern != "empty":
        assert bool(result.abs().any())


@pytest.mark.parametrize("bad", ["negative_index", "out_of_range", "nonmonotone", "wrong_terminal", "wrong_dtype"])
def test_csr_guards(bad: str) -> None:
    inputs = make_inputs(Case("guard", 12, 2, 16, 8, 8, "prefix_only"))
    if bad == "negative_index":
        inputs["kv_indices_prefix"][0] = -1
    elif bad == "out_of_range":
        inputs["kv_indices_prefix"][0] = 8
    elif bad == "nonmonotone":
        inputs["kv_indptr_prefix"][1] = -1
    elif bad == "wrong_terminal":
        inputs["kv_indptr_prefix"][-1] += 1
    else:
        inputs["kv_indices_prefix"] = inputs["kv_indices_prefix"].to(torch.int64)
    with pytest.raises(ValueError):
        validate_inputs(inputs)
