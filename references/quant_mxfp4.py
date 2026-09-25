"""Independent byte-level oracle for the no-shuffle MXFP4 Even-mode task.

The oracle intentionally uses scalar CPU arithmetic and explicit E2M1 levels,
not AITER's bit-parallel quantization code or its Python test reference.
"""

from __future__ import annotations

import math
import random
import struct
import ctypes
from dataclasses import dataclass


LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GUARD_BYTES = 256
GUARD_VALUE = 0xA5


@dataclass
class GuardedOutputs:
    packed: object
    scales: object
    packed_storage: object
    scale_storage: object

    def __iter__(self):
        yield self.packed
        yield self.scales

    def __getitem__(self, index):
        return (self.packed, self.scales)[index]

    def check_guards(self) -> None:
        import torch

        for name, storage in (("packed", self.packed_storage), ("scale", self.scale_storage)):
            if not bool(torch.all(storage[:GUARD_BYTES] == GUARD_VALUE)) or not bool(
                torch.all(storage[-GUARD_BYTES:] == GUARD_VALUE)
            ):
                raise ValueError(f"{name} output guard was modified")


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_float32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def _scale_even(max_abs: float) -> tuple[float, int]:
    # Matches the documented Even-mode scale rule: round amax to a power of
    # two at the 1.75 threshold, then divide by four and clamp E8M0 range.
    bits = (_float32_bits(max_abs) + 0x00200000) & 0x7F800000
    rounded = _bits_float32(bits)
    exponent = -127 if rounded == 0.0 else max(-127, min(127, math.floor(math.log2(rounded)) - 2))
    scale = math.ldexp(1.0, exponent)
    return scale, exponent + 127


def _fp4_code(value: float) -> int:
    magnitude = abs(value)
    # Adjacent E2M1 levels are an integer sequence of codes; even code wins ties.
    best = min(range(8), key=lambda code: (abs(magnitude - LEVELS[code]), code & 1))
    return best | (8 if math.copysign(1.0, value) < 0 else 0)


def oracle(x) -> tuple[bytes, bytes]:
    """Return packed FP4 and E8M0 bytes for a 2D torch input tensor."""
    rows, cols = x.shape
    if cols % 32:
        raise ValueError("the task requires 32-element quantization groups")
    values = x.cpu().float().tolist()
    packed = bytearray(rows * cols // 2)
    scales = bytearray(rows * cols // 32)
    for row in range(rows):
        for group in range(cols // 32):
            start = group * 32
            chunk = values[row][start : start + 32]
            scale, encoded = _scale_even(max(abs(value) for value in chunk))
            scales[row * (cols // 32) + group] = encoded
            for offset in range(0, 32, 2):
                lo = _fp4_code(chunk[offset] / scale)
                hi = _fp4_code(chunk[offset + 1] / scale)
                packed[row * (cols // 2) + start // 2 + offset // 2] = lo | (hi << 4)
    return bytes(packed), bytes(scales)


def make_input(case: dict):
    import torch

    rows, cols = int(case["rows"]), int(case["cols"])
    if rows <= 0 or cols <= 0 or cols % 32:
        raise ValueError("invalid MXFP4 case shape")
    rng = random.Random(int(case["seed"]))
    values = [rng.gauss(0, float(case.get("std", 0.75))) for _ in range(rows * cols)]
    if case.get("edges"):
        values[: min(len(values), 16)] = [
            0.0, -0.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25,
            1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.0, -5.0,
        ][: min(len(values), 16)]
    if case.get("zero_group"):
        values[:32] = [0.0] * 32
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[case["dtype"]]
    return torch.tensor(values, dtype=dtype, device="cpu").reshape(rows, cols).to("cuda")


def run_aiter(x):
    from aiter.ops.quant import quant_mxfp4

    output = allocate_outputs(x)
    quant_mxfp4(x, output.packed, output.scales, 32, 2, False, False, False, False)
    return output


def allocate_outputs(x):
    import torch

    rows, cols = x.shape
    packed_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    scale_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
    packed_count, scale_count = rows * cols // 2, rows * cols // 32
    packed_storage = torch.full(
        (GUARD_BYTES + packed_count + GUARD_BYTES,), GUARD_VALUE,
        device=x.device, dtype=torch.uint8,
    )
    scale_storage = torch.full(
        (GUARD_BYTES + scale_count + GUARD_BYTES,), GUARD_VALUE,
        device=x.device, dtype=torch.uint8,
    )
    return GuardedOutputs(
        packed_storage[GUARD_BYTES : GUARD_BYTES + packed_count].view(packed_dtype).reshape(rows, cols // 2),
        scale_storage[GUARD_BYTES : GUARD_BYTES + scale_count].view(scale_dtype).reshape(rows, cols // 32),
        packed_storage,
        scale_storage,
    )


def output_bytes(output) -> tuple[bytes, bytes]:
    import torch

    packed, scales = output
    return (
        bytes(packed.contiguous().view(torch.uint8).cpu().reshape(-1).tolist()),
        bytes(scales.contiguous().view(torch.uint8).cpu().reshape(-1).tolist()),
    )


def validate_output(output, x, aiter_output=None) -> None:
    packed, scales = output
    rows, cols = x.shape
    if packed.shape != (rows, cols // 2) or scales.shape != (rows, cols // 32):
        raise ValueError("output shapes do not match the operator contract")
    if not packed.is_contiguous() or not scales.is_contiguous():
        raise ValueError("output layout is not contiguous")
    if packed.device != x.device or scales.device != x.device:
        raise ValueError("outputs must be on the input GPU")
    if aiter_output is not None and (
        packed.dtype != aiter_output[0].dtype or scales.dtype != aiter_output[1].dtype
    ):
        raise ValueError("output dtypes differ from the pinned AITER operator")
    output.check_guards()


def load_hip_candidate(path):
    library = ctypes.CDLL(str(path))
    launch = library.aiter_rs_quant_mxfp4_even
    launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int

    class Candidate:
        def run_into(self, x, output):
            import torch

            rows, cols = x.shape
            dtype = 0 if x.dtype == torch.float16 else 1
            status = launch(
                ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(output.packed.data_ptr()),
                ctypes.c_void_p(output.scales.data_ptr()), rows, cols, dtype,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            )
            if status:
                raise RuntimeError(f"HIP launch failed with error {status}")
            return output

        def run(self, x):
            return self.run_into(x, allocate_outputs(x))

    return Candidate()


def benchmark_calls(x, candidate):
    from aiter.ops.quant import quant_mxfp4

    baseline_output = allocate_outputs(x)
    candidate_output = allocate_outputs(x)

    def baseline():
        quant_mxfp4(x, baseline_output.packed, baseline_output.scales, 32, 2, False, False, False, False)

    def proposed():
        candidate.run_into(x, candidate_output)

    def check_guards():
        baseline_output.check_guards()
        candidate_output.check_guards()

    return baseline, proposed, check_guards
