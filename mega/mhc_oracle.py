"""CPU mathematical oracle for the four-output mHC post/pre operator."""

from __future__ import annotations

from dataclasses import dataclass


HC_MULT = 4
HC_MIXES = HC_MULT * (HC_MULT + 2)
FUSED_M_BOUND = 1024


@dataclass(frozen=True)
class Parameters:
    rms_eps: float = 1e-6
    pre_eps: float = 1e-6
    sinkhorn_eps: float = 1e-6
    post_multiplier: float = 2.0
    sinkhorn_repeat: int = 20
    norm_eps: float = 1e-6


def validate_case(case: dict) -> None:
    m, hidden = int(case["m"]), int(case["hidden_size"])
    if m <= 0 or hidden < 512 or hidden % 256:
        raise ValueError("m must be positive and hidden_size must be >=512 and divisible by 256")
    if case["route"] not in ("fused", "large_m_fallback"):
        raise ValueError("unknown MHC route")
    expected = "fused" if m <= FUSED_M_BOUND else "large_m_fallback"
    if case["route"] != expected:
        raise ValueError("case route does not match gfx950 force_fused dispatch")
    if case.get("norm") and hidden not in (1280, 2560, 4096, 5120, 7168):
        raise ValueError("RMSNorm reduction supports hidden sizes 1280, 2560, 4096, 5120, 7168")
    if case["route"] == "large_m_fallback":
        residual_block = 1024 if hidden % 1024 == 0 else 512 if hidden % 512 == 0 else 256
        if hidden < 2 * residual_block:
            raise ValueError("large-M post requires two residual blocks for prefetch")
    if not isinstance(case["seed"], int):
        raise ValueError("seed must be an integer")


def make_inputs(case: dict, device: str = "cpu") -> dict:
    import torch

    validate_case(case)
    m, hidden = case["m"], case["hidden_size"]
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])

    def rand(shape, std):
        return torch.randn(shape, dtype=torch.float32, generator=generator) * std

    values = {
        "layer_input": rand((m, hidden), 0.15).to(torch.bfloat16),
        "residual_in": rand((m, HC_MULT, hidden), 0.15).to(torch.bfloat16),
        "post_layer_mix": rand((m, HC_MULT, 1), 0.1),
        "comb_res_mix": rand((m, HC_MULT, HC_MULT), 0.1),
        "fn": rand((HC_MIXES, HC_MULT * hidden), 0.03),
        "hc_scale": torch.tensor([0.08, 0.11, 0.09], dtype=torch.float32),
        "hc_base": rand((HC_MIXES,), 0.06),
    }
    if case.get("post_rank2", False):
        values["post_layer_mix"] = values["post_layer_mix"].squeeze(-1)
    if case.get("norm", False):
        values["norm_weight"] = (1.0 + rand((hidden,), 0.04)).to(torch.bfloat16)
    return {name: value.to(device) for name, value in values.items()}


def post_pre(inputs: dict, params: Parameters = Parameters()) -> tuple:
    """Return (post_mix, comb_mix, layer_input_out, next_residual)."""
    import torch

    x = inputs["layer_input"].float()
    residual = inputs["residual_in"].float()
    post = inputs["post_layer_mix"].reshape(x.shape[0], HC_MULT).float()
    combination = inputs["comb_res_mix"].float()

    # comb_res_mix[input_head, output_head] multiplies the old residual.
    mixed_residual = torch.einsum("mij,mih->mjh", combination, residual)
    next_residual = (mixed_residual + post[:, :, None] * x[:, None, :]).to(torch.bfloat16)

    flat = next_residual.float().reshape(x.shape[0], -1)
    rms_inverse = torch.rsqrt(flat.square().mean(dim=1) + params.rms_eps)
    projections = torch.matmul(flat, inputs["fn"].float().T) * rms_inverse[:, None]
    scale = inputs["hc_scale"].float()
    base = inputs["hc_base"].float()

    pre_logits = projections[:, :HC_MULT] * scale[0] + base[:HC_MULT]
    post_logits = projections[:, HC_MULT : 2 * HC_MULT] * scale[1] + base[HC_MULT : 2 * HC_MULT]
    comb_logits = (
        projections[:, 2 * HC_MULT :] * scale[2] + base[2 * HC_MULT :]
    ).reshape(-1, HC_MULT, HC_MULT)

    pre_mix = torch.sigmoid(pre_logits) + params.pre_eps
    post_mix = (torch.sigmoid(post_logits) * params.post_multiplier).unsqueeze(-1)
    comb_mix = torch.softmax(comb_logits, dim=-1) + params.sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + params.sinkhorn_eps)
    for _ in range(params.sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + params.sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + params.sinkhorn_eps)

    layer_output = (next_residual.float() * pre_mix[:, :, None]).sum(dim=1)
    norm_weight = inputs.get("norm_weight")
    if norm_weight is not None:
        norm_inverse = torch.rsqrt(layer_output.square().mean(dim=-1, keepdim=True) + params.norm_eps)
        layer_output = layer_output * norm_inverse * norm_weight.float()
    return post_mix, comb_mix, layer_output.to(torch.bfloat16), next_residual
