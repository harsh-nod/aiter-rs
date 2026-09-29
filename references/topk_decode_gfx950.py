"""Independent stable decode top-k oracle and gfx950 admission cases.

This module does not import AITER. GPU dispatch is inspected only by the
separate, trusted admission probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


WIDTH = 65_536
K = 512
ROWS = 4
Kind = Literal["random", "ties", "padded", "next_n", "signed_zero"]


@dataclass(frozen=True)
class Case:
    name: str
    kind: Kind
    seed: int
    seq_lens: tuple[int, ...]
    next_n: int = 1
    width: int = WIDTH
    rows: int = ROWS
    k: int = K

    def validate(self) -> None:
        if self.rows < 1 or self.width < self.k or self.k < 1 or self.next_n < 1:
            raise ValueError("invalid top-k shape")
        if len(self.seq_lens) < (self.rows + self.next_n - 1) // self.next_n:
            raise ValueError("seq_lens has too few entries")
        if any(not 0 < self.effective_length(row) <= self.width for row in range(self.rows)):
            raise ValueError("effective lengths must fit the physical row")

    def effective_length(self, row: int) -> int:
        return self.seq_lens[row // self.next_n] - self.next_n + row % self.next_n + 1


def public_cases() -> tuple[Case, ...]:
    return (
        Case("random_full", "random", 1031, (WIDTH,) * ROWS),
        Case("heavy_ties", "ties", 2077, (WIDTH,) * ROWS),
        Case("short_padded", "padded", 3109, (130, 511, 1024, WIDTH)),
        Case("decode_next_n_2", "next_n", 4099, (50_000, WIDTH), next_n=2),
        Case("signed_zero", "signed_zero", 5051, (WIDTH,) * ROWS),
    )


def make_logits(case: Case) -> torch.Tensor:
    case.validate()
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    values = torch.randn((case.rows, case.width), generator=generator, dtype=torch.float32)
    if case.kind == "ties":
        values = (values * 4).round() / 4
    elif case.kind == "signed_zero":
        values.zero_()
        values.view(torch.int32)[:, ::2] = -(1 << 31)
    elif case.kind == "padded":
        for row in range(case.rows):
            values[row, case.effective_length(row) :] = 1_000_000.0
    return values


def stable_topk_indices(row: torch.Tensor, k: int) -> list[int]:
    """Rank raw float32 bits with index ties; emit selected indices in order."""
    if row.device.type != "cpu" or row.dtype != torch.float32 or row.ndim != 1:
        raise ValueError("expected a CPU float32 row")
    bits = row.contiguous().view(torch.int32).tolist()

    def key(index: int) -> tuple[int, int]:
        raw = bits[index] & 0xFFFFFFFF
        mask = 0 if raw >> 31 else 0x7FFFFFFF
        return raw ^ mask, index

    chosen = sorted(range(len(bits)), key=key)[: min(k, len(bits))]
    return sorted(chosen) + [-1] * max(0, k - len(bits))


def oracle(case: Case, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    case.validate()
    if logits.shape != (case.rows, case.width) or logits.device.type != "cpu":
        raise ValueError("logits must be a CPU tensor with the case shape")
    indices = torch.empty((case.rows, case.k), dtype=torch.int32)
    values = torch.full((case.rows, case.k), float("-inf"), dtype=torch.float32)
    for row in range(case.rows):
        selected = stable_topk_indices(logits[row, : case.effective_length(row)], case.k)
        indices[row] = torch.tensor(selected, dtype=torch.int32)
        for col, index in enumerate(selected):
            if index >= 0:
                values[row, col] = logits[row, index]
    return indices, values
