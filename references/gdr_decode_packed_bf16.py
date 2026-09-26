"""Independent CPU oracle and guarded gfx950 GDR decode admission cases."""

from __future__ import annotations

from dataclasses import dataclass


Q_HEADS = 8
V_HEADS = 32
K = 128
V = 128
QKV_DIM = 2 * Q_HEADS * K + V_HEADS * V
STATE_SLOT_ELEMENTS = V_HEADS * V * K
SCALE = K**-0.5
RTOL = 1.0e-2
ATOL = 1.0e-3
GUARD_BYTES = 256
GUARD_VALUE = 0xA5


def validate_case(case: dict) -> None:
    batch, pool, steps = (int(case[key]) for key in ("batch", "pool", "steps"))
    indices = case["indices"]
    if batch <= 0 or pool <= 0 or steps <= 0 or len(indices) != batch:
        raise ValueError("GDR case needs positive batch, pool and steps, and one index per row")
    if any(not isinstance(index, int) or index < -(2**31) or index >= 2**31 for index in indices):
        raise ValueError("indices must be signed INT32 values")
    valid = [index for index in indices if 0 <= index < pool]
    if len(valid) != len(set(valid)):
        raise ValueError("duplicate valid state indices are outside the AITER contract")
    if case.get("indices_stride", 1) not in (1, 2):
        raise ValueError("pilot supports index strides 1 or 2")


def _sample(shape, std: float, generator):
    import torch

    return (torch.randn(shape, generator=generator, dtype=torch.float32) * std).to(torch.bfloat16)


def make_initial_state(case: dict):
    import torch

    validate_case(case)
    generator = torch.Generator(device="cpu").manual_seed(int(case["seed"]) ^ 0x5A173D)
    return _sample((case["pool"], V_HEADS, V, K), 0.02, generator)


def make_step_inputs(case: dict, step: int) -> dict:
    import torch

    validate_case(case)
    if not 0 <= step < case["steps"]:
        raise ValueError("step is outside the case")
    generator = torch.Generator(device="cpu").manual_seed(int(case["seed"]) + step * 131)
    batch = case["batch"]
    return {
        "mixed_qkv": _sample((batch, QKV_DIM), 0.1, generator),
        "a": _sample((batch, V_HEADS), 0.2, generator),
        "b": _sample((batch, V_HEADS), 0.2, generator),
        "dt_bias": _sample((V_HEADS,), 0.1, generator),
        "A_log": -3.0 + 2.0 * torch.rand((V_HEADS,), generator=generator),
    }


def oracle_step(inputs: dict, indices: list[int], state):
    """Mutate CPU BF16 state exactly once per valid row; return CPU BF16 output."""
    import torch
    from torch.nn import functional as F

    batch = inputs["mixed_qkv"].shape[0]
    if state.device.type != "cpu" or state.dtype != torch.bfloat16:
        raise ValueError("oracle state must be CPU BF16")
    if len(indices) != batch:
        raise ValueError("indices/batch mismatch")
    valid = [index for index in indices if 0 <= index < state.shape[0]]
    if len(valid) != len(set(valid)):
        raise ValueError("duplicate valid state indices are undefined")

    out = torch.zeros((batch, 1, V_HEADS, V), dtype=torch.bfloat16)
    for row, slot in enumerate(indices):
        if not 0 <= slot < state.shape[0]:
            continue
        packed = inputs["mixed_qkv"][row].float()
        q = packed[: Q_HEADS * K].reshape(Q_HEADS, K)
        key = packed[Q_HEADS * K : 2 * Q_HEADS * K].reshape(Q_HEADS, K)
        value = packed[2 * Q_HEADS * K :].reshape(V_HEADS, V)

        q = q / torch.sqrt(torch.sum(q * q, dim=1, keepdim=True) + 1.0e-6) * SCALE
        key = key / torch.sqrt(torch.sum(key * key, dim=1, keepdim=True) + 1.0e-6)
        q = torch.repeat_interleave(q, V_HEADS // Q_HEADS, dim=0)
        key = torch.repeat_interleave(key, V_HEADS // Q_HEADS, dim=0)

        gate = F.softplus(inputs["a"][row].float() + inputs["dt_bias"].float())
        decay = torch.exp(-torch.exp(inputs["A_log"].float()) * gate)
        beta = torch.sigmoid(inputs["b"][row].float()).to(torch.bfloat16).float()
        recurrent = state[slot].float() * decay[:, None, None]
        h_key = torch.einsum("hvk,hk->hv", recurrent, key)
        h_query = torch.einsum("hvk,hk->hv", recurrent, q)
        key_query = torch.sum(key * q, dim=1)
        residual = (value - h_key) * beta[:, None]
        out[row, 0] = (h_query + residual * key_query[:, None]).to(torch.bfloat16)
        state[slot] = (recurrent + residual[:, :, None] * key[:, None, :]).to(torch.bfloat16)
    return out


@dataclass
class GuardedState:
    tensor: object
    storage: object
    slot_bytes: int
    stride_bytes: int
    pool: int

    def check(self) -> None:
        import torch

        if not bool(torch.all(self.storage[:GUARD_BYTES] == GUARD_VALUE)):
            raise ValueError("state prefix guard modified")
        for slot in range(self.pool):
            start = GUARD_BYTES + slot * self.stride_bytes + self.slot_bytes
            end = GUARD_BYTES + (slot + 1) * self.stride_bytes
            if not bool(torch.all(self.storage[start:end] == GUARD_VALUE)):
                raise ValueError("state slot gap modified")
        if not bool(torch.all(self.storage[-GUARD_BYTES:] == GUARD_VALUE)):
            raise ValueError("state suffix guard modified")


@dataclass
class GuardedOutput:
    tensor: object
    storage: object

    def check(self) -> None:
        import torch

        if not bool(torch.all(self.storage[:GUARD_BYTES] == GUARD_VALUE)) or not bool(
            torch.all(self.storage[-GUARD_BYTES:] == GUARD_VALUE)
        ):
            raise ValueError("output guard modified")


def guarded_state(cpu_state, *, slot_padding: bool) -> GuardedState:
    import torch

    pool = cpu_state.shape[0]
    slot_bytes = STATE_SLOT_ELEMENTS * 2
    stride_bytes = slot_bytes + (GUARD_BYTES if slot_padding else 0)
    storage = torch.full(
        (GUARD_BYTES + pool * stride_bytes + GUARD_BYTES,),
        GUARD_VALUE, dtype=torch.uint8, device="cuda",
    )
    body = storage[GUARD_BYTES : GUARD_BYTES + pool * stride_bytes].view(torch.bfloat16)
    stride_elements = stride_bytes // 2
    view = torch.as_strided(
        body, (pool, V_HEADS, V, K), (stride_elements, V * K, K, 1)
    )
    view.copy_(cpu_state.to("cuda"))
    return GuardedState(view, storage, slot_bytes, stride_bytes, pool)


def guarded_output(batch: int) -> GuardedOutput:
    import torch

    bytes_needed = batch * V_HEADS * V * 2
    storage = torch.full(
        (GUARD_BYTES + bytes_needed + GUARD_BYTES,),
        GUARD_VALUE, dtype=torch.uint8, device="cuda",
    )
    out = storage[GUARD_BYTES : GUARD_BYTES + bytes_needed].view(torch.bfloat16)
    return GuardedOutput(out.reshape(batch, 1, V_HEADS, V), storage)


def _padded_rows(cpu_tensor):
    import torch

    rows, cols = cpu_tensor.shape
    storage = torch.full((rows, 2, cols), -13.0, dtype=cpu_tensor.dtype, device="cuda")
    view = storage[:, 0, :]
    view.copy_(cpu_tensor.to("cuda"))
    return view, storage[:, 1, :]


def gpu_step_inputs(cpu_inputs: dict, case: dict) -> tuple[dict, list]:
    import torch

    tensors, padding = {}, []
    for name in ("mixed_qkv", "a", "b"):
        if case.get("strided_rows"):
            tensors[name], guard = _padded_rows(cpu_inputs[name])
            padding.append(guard)
        else:
            tensors[name] = cpu_inputs[name].to("cuda")
    for name in ("dt_bias", "A_log"):
        tensors[name] = cpu_inputs[name].to("cuda")
    values = torch.tensor(case["indices"], dtype=torch.int32, device="cuda")
    if case.get("indices_stride", 1) == 2:
        storage = torch.full((case["batch"] * 2,), -777, dtype=torch.int32, device="cuda")
        storage[::2] = values
        tensors["indices"] = storage[::2]
        padding.append(storage[1::2])
    else:
        tensors["indices"] = values
    return tensors, padding


def run_aiter(inputs: dict, state, out):
    from aiter import gdr_decode_packed_bf16

    return gdr_decode_packed_bf16(
        inputs["mixed_qkv"], inputs["a"], inputs["b"], inputs["dt_bias"],
        inputs["A_log"], inputs["indices"], state, out, scale=SCALE,
    )


def compare_step(
    cpu_inputs: dict, indices: list[int], expected_out, expected_state,
    gpu_inputs: dict, gpu_state: GuardedState, gpu_out: GuardedOutput, padding: list,
) -> None:
    import torch

    actual_out = gpu_out.tensor.cpu()
    actual_state = gpu_state.tensor.cpu()
    torch.testing.assert_close(actual_out.float(), expected_out.float(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual_state.float(), expected_state.float(), rtol=RTOL, atol=ATOL)

    pool = expected_state.shape[0]
    valid = {index for index in indices if 0 <= index < pool}
    invalid_rows = [row for row, index in enumerate(indices) if not 0 <= index < pool]
    if invalid_rows and not bool(torch.all(actual_out[invalid_rows].view(torch.uint16) == 0)):
        raise AssertionError("invalid rows must be positive BF16 zero")
    untouched = [slot for slot in range(pool) if slot not in valid]
    if untouched and not torch.equal(
        actual_state[untouched].view(torch.uint16), expected_state[untouched].view(torch.uint16)
    ):
        raise AssertionError("untouched state slots changed")
    for name, expected in cpu_inputs.items():
        if not torch.equal(gpu_inputs[name].cpu(), expected):
            raise AssertionError(f"input {name} changed")
    if not torch.equal(gpu_inputs["indices"].cpu(), torch.tensor(indices, dtype=torch.int32)):
        raise AssertionError("indices changed")
    for guard in padding:
        if guard.dtype == torch.int32:
            intact = bool(torch.all(guard == -777))
        else:
            intact = bool(torch.all(guard == -13.0))
        if not intact:
            raise AssertionError("input padding changed")
    gpu_state.check()
    gpu_out.check()
