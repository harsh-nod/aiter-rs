"""CPU oracle and public BF16 CSR cases for gfx950 sparse prefill admission.

This module has no AITER import. The operation is two-source sparse attention
with a denominator-only, per-head sink.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


AITER_SHA = "868ccf62a0bcad3aa47f92728340ccb37ed4fb39"
D = 512
Pattern = Literal["mixed", "tile_tail", "prefix_only", "extend_only", "empty", "dense"]


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    n: int
    h: int
    pages: int
    extend_rows: int
    pattern: Pattern
    softmax_scale: float = 1 / math.sqrt(D)

    def validate(self) -> None:
        if not self.name or self.seed < 0 or self.n < 1 or not 1 <= self.h <= 32:
            raise ValueError("invalid case identity or query dimensions")
        if self.pages < 1 or self.extend_rows < 1:
            raise ValueError("both KV source allocations must be nonempty")
        if self.pattern not in ("mixed", "tile_tail", "prefix_only", "extend_only", "empty", "dense"):
            raise ValueError("unknown CSR pattern")
        if self.pattern == "tile_tail" and (self.pages < 65 or self.extend_rows < 65):
            raise ValueError("tile-tail case requires at least 65 rows per source")
        if not math.isfinite(self.softmax_scale):
            raise ValueError("softmax_scale must be finite")


def public_cases() -> tuple[Case, ...]:
    return (
        Case("mixed_sources", 1103, 4, 16, 80, 80, "mixed"),
        Case("tile_tail_63_64_65", 2207, 3, 32, 80, 80, "tile_tail"),
        Case("prefix_only", 3301, 3, 16, 80, 80, "prefix_only"),
        Case("extend_only", 4409, 3, 16, 80, 80, "extend_only"),
        Case("sink_only", 5501, 2, 16, 80, 80, "empty"),
    )


def _row_lengths(case: Case) -> tuple[list[int], list[int]]:
    if case.pattern == "mixed":
        prefix = [min(case.pages, (i * 17 + 3) % 67) for i in range(case.n)]
        extend = [min(case.extend_rows, (i * 19 + 5) % 67) for i in range(case.n)]
        prefix[0], extend[0] = 0, 0
    elif case.pattern == "tile_tail":
        prefix = [63, 64, 65] * ((case.n + 2) // 3)
        extend = [1, 0, 65] * ((case.n + 2) // 3)
    elif case.pattern == "prefix_only":
        prefix = [min(case.pages, i * 7 + 1) for i in range(case.n)]
        extend = [0] * case.n
    elif case.pattern == "extend_only":
        prefix = [0] * case.n
        extend = [min(case.extend_rows, i * 7 + 1) for i in range(case.n)]
    elif case.pattern == "empty":
        prefix = extend = [0] * case.n
    else:
        prefix = [case.pages] * case.n
        extend = [case.extend_rows] * case.n
    return prefix[: case.n], extend[: case.n]


def _csr(lengths: list[int], rows: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    indptr = [0]
    indices: list[int] = []
    for length in lengths:
        if not 0 <= length <= rows:
            raise ValueError("CSR row length exceeds allocated rows")
        if length:
            indices.extend(torch.randperm(rows, generator=generator)[:length].tolist())
        indptr.append(len(indices))
    return torch.tensor(indptr, dtype=torch.int32), torch.tensor(indices, dtype=torch.int32)


def make_inputs(case: Case) -> dict[str, torch.Tensor]:
    case.validate()
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    prefix_lengths, extend_lengths = _row_lengths(case)
    ip_p, ix_p = _csr(prefix_lengths, case.pages, generator)
    ip_e, ix_e = _csr(extend_lengths, case.extend_rows, generator)
    tensors = {
        "q": (torch.randn((case.n, case.h, D), generator=generator) * 0.5).to(torch.bfloat16),
        "unified_kv": (torch.randn((case.pages, D), generator=generator) * 0.5).to(torch.bfloat16),
        "kv_indices_prefix": ix_p,
        "kv_indptr_prefix": ip_p,
        "kv": (torch.randn((case.extend_rows, D), generator=generator) * 0.5).to(torch.bfloat16),
        "kv_indices_extend": ix_e,
        "kv_indptr_extend": ip_e,
        "attn_sink": torch.randn((case.h,), generator=generator, dtype=torch.float32) * 0.25,
    }
    validate_inputs(tensors)
    return tensors


def validate_inputs(inputs: dict[str, torch.Tensor]) -> None:
    required = {
        "q", "unified_kv", "kv_indices_prefix", "kv_indptr_prefix", "kv",
        "kv_indices_extend", "kv_indptr_extend", "attn_sink",
    }
    if set(inputs) != required:
        raise ValueError("unexpected or missing sparse-prefill input")
    q, prefix, extend = (inputs[key] for key in ("q", "unified_kv", "kv"))
    if q.ndim != 3 or q.shape[-1] != D or q.shape[0] < 1 or not 1 <= q.shape[1] <= 32:
        raise ValueError("q must have shape [N,H,512] with N>0 and H<=32")
    if q.dtype != torch.bfloat16 or q.device.type != "cpu":
        raise ValueError("oracle requires CPU BF16 q")
    for tensor in (q, prefix, extend):
        if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("Q and both KV sources must be contiguous CPU BF16")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Q and both KV sources must be finite")
    if prefix.ndim != 2 or extend.ndim != 2 or prefix.shape[1] != D or extend.shape[1] != D:
        raise ValueError("KV sources must have shape [rows,512]")
    if prefix.shape[0] < 1 or extend.shape[0] < 1:
        raise ValueError("both KV source allocations must be nonempty")
    sink = inputs["attn_sink"]
    if sink.shape != (q.shape[1],) or sink.dtype != torch.float32 or sink.device.type != "cpu":
        raise ValueError("attn_sink must be CPU FP32 [H]")
    if not sink.is_contiguous() or not bool(torch.isfinite(sink).all()):
        raise ValueError("attn_sink must be contiguous and finite")
    for suffix, rows in (("prefix", prefix.shape[0]), ("extend", extend.shape[0])):
        indptr = inputs[f"kv_indptr_{suffix}"]
        indices = inputs[f"kv_indices_{suffix}"]
        if indptr.device.type != "cpu" or indices.device.type != "cpu":
            raise ValueError("CSR arrays must be on CPU")
        if indptr.dtype != torch.int32 or indices.dtype != torch.int32:
            raise ValueError("CSR arrays must be int32")
        if not indptr.is_contiguous() or not indices.is_contiguous():
            raise ValueError("CSR arrays must be contiguous")
        if indptr.shape != (q.shape[0] + 1,) or indices.ndim != 1:
            raise ValueError("CSR pointer or index shape mismatch")
        if indptr[0].item() != 0 or indptr[-1].item() != indices.numel():
            raise ValueError("CSR pointers must cover exactly the index array")
        if bool((indptr[1:] < indptr[:-1]).any()):
            raise ValueError("CSR pointers must be monotonic")
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= rows):
            raise ValueError("CSR index out of bounds")


def oracle(inputs: dict[str, torch.Tensor], softmax_scale: float) -> torch.Tensor:
    validate_inputs(inputs)
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")
    q = inputs["q"].float()
    prefix = inputs["unified_kv"].float()
    extend = inputs["kv"].float()
    n, h, _ = q.shape
    result = torch.zeros((n, h, D), dtype=torch.bfloat16)
    for token in range(n):
        rows = []
        for suffix, kv in (("prefix", prefix), ("extend", extend)):
            indptr = inputs[f"kv_indptr_{suffix}"]
            indices = inputs[f"kv_indices_{suffix}"]
            start, end = int(indptr[token]), int(indptr[token + 1])
            if end > start:
                rows.append(kv.index_select(0, indices[start:end].to(torch.int64)))
        if not rows:
            continue
        keys = torch.cat(rows, dim=0)
        scores = (q[token] @ keys.T) * softmax_scale
        sink = inputs["attn_sink"][:, None]
        maximum = torch.maximum(scores.max(dim=1, keepdim=True).values, sink)
        numerator = torch.exp(scores - maximum)
        denominator = numerator.sum(dim=1, keepdim=True) + torch.exp(sink - maximum)
        result[token] = ((numerator / denominator) @ keys).to(torch.bfloat16)
    return result
