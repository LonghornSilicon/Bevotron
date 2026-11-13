"""Simulated post-training quantization.

Values are rounded to an integer grid and scaled back to float ("fake quant"), so the
model runs in float but only ever sees numbers an int8/int4 chip could represent.
That is enough to measure the accuracy cost; real integer kernels live in the
inference engine.

  weights      symmetric, one scale per output channel, round-to-nearest
  activations  symmetric, applied to every Linear input; one scale per token (dynamic),
               one fixed scale per Linear (static, quant.scales), or one sampled from
               part of the matrix (first_row, first_n, row_first_n; quant.hybrid)
  KV cache     symmetric, applied as K and V are produced; one scale per token per head
               (dynamic), one fixed scale per head (static), or sampled per head
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bevotron.quant.hybrid import (
    SampledKV,
    ScaleMemory,
    fake_quantize_sampled,
    sampling_of,
    start_margins,
)
from bevotron.quant.static import StaticKV, calibrate, calibration_batches, fake_quantize_static

# scheme -> bits for (weights, activations, KV cache); None = leave in float
SCHEME_BITS: dict[str, tuple[int | None, int | None, int | None]] = {
    "none": (None, None, None),
    "int8": (8, 8, 8),
    "w4a8": (4, 8, 8),
    "w4a4": (4, 4, 4),
    "w4a4kv4": (4, 4, 4),
    "w4a8kv4": (4, 8, 4),
}


def fake_quantize(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Round x to a symmetric `bits`-bit grid, one scale per slice reduced along `dim`.

    Weights: dim=1 gives one scale per output channel. Activations (B, T, C) and the
    KV cache (B, heads, T, head_dim): dim=-1 gives one scale per token (per head).
    """
    qmax = 2 ** (bits - 1) - 1
    amax = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale


def apply_ptq(
    model: nn.Module, scheme: str, scales: dict | None = None, sampling: tuple | None = None
) -> nn.Module:
    """Quantize a trained model in place according to `scheme`. With `scales` (log scales
    from QAT or calibration, see quant.static) activations and the KV cache use those fixed
    scales; without, every token gets its own. With `sampling` = (mode, n), `scales` holds
    log margins and each matrix's scale is sampled from it (quant.hybrid), carried across
    the calls of one sequence so cached generation reuses the first rows' scale."""
    w_bits, a_bits, kv_bits = SCHEME_BITS[scheme]
    memory = None
    if sampling is not None:
        memory = model.scale_memory = ScaleMemory()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.dim() == 2:
            if w_bits is not None:
                with torch.no_grad():
                    module.weight.copy_(fake_quantize(module.weight, w_bits, dim=1))
            if a_bits is not None:
                if scales is None:
                    module.register_forward_pre_hook(_activation_hook(a_bits))
                elif sampling is not None:
                    margin = scales[f"{name}.act_log_margin"].to(module.weight.device)
                    module.register_forward_pre_hook(
                        _sampled_activation_hook(a_bits, sampling, margin, memory, name)
                    )
                else:
                    log_scale = scales[f"{name}.act_log_scale"].to(module.weight.device)
                    module.register_forward_pre_hook(_static_activation_hook(a_bits, log_scale))
        if kv_bits is not None and hasattr(module, "kv_quant"):
            if scales is None:
                module.kv_quant = lambda k, v, b=kv_bits: (
                    fake_quantize(k, b, dim=-1),  # per token, per head
                    fake_quantize(v, b, dim=-1),
                )
            elif sampling is not None:
                device = next(module.parameters()).device
                module.kv_quant = SampledKV(
                    kv_bits,
                    *sampling,
                    scales[f"{name}.kv_quant.k_log_margin"].to(device),
                    scales[f"{name}.kv_quant.v_log_margin"].to(device),
                    memory,
                    name,
                ).requires_grad_(False)
            else:
                device = next(module.parameters()).device
                module.kv_quant = StaticKV(
                    kv_bits,
                    scales[f"{name}.kv_quant.k_log_scale"].to(device),
                    scales[f"{name}.kv_quant.v_log_scale"].to(device),
                ).requires_grad_(False)
    return model


def quantize_for_eval(model: nn.Module, config, scales: dict, device: str) -> nn.Module:
    """Apply the run's scheme to a loaded model: learned static scales if the checkpoint
    has them, calibrated ones for a static PTQ run, sampled scales with the learned (or
    configured) margins for a sampled run, per-token scales otherwise."""
    scheme = config.quant.scheme
    if scheme == "none":
        return model
    sampling = sampling_of(config.quant)
    if sampling is not None:
        _, a_bits, kv_bits = SCHEME_BITS[scheme]
        margins = scales or start_margins(model, config.quant.scale_margin, a_bits, kv_bits)
        return apply_ptq(model, scheme, margins, sampling)
    if config.quant.scales != "static":
        return apply_ptq(model, scheme)
    if not scales:
        _, a_bits, kv_bits = SCHEME_BITS[scheme]
        scales = calibrate(model, calibration_batches(config, device), a_bits, kv_bits)
    return apply_ptq(model, scheme, scales)


def _activation_hook(bits: int):
    def hook(_module, inputs):
        x = inputs[0]
        return (fake_quantize(x, bits, dim=-1),) + tuple(inputs[1:])

    return hook


def _sampled_activation_hook(
    bits: int, sampling: tuple, log_margin: torch.Tensor, memory: ScaleMemory, key: str
):
    def hook(_module, inputs):
        x = fake_quantize_sampled(inputs[0], bits, *sampling, log_margin, memory, key)
        return (x,) + tuple(inputs[1:])

    return hook


def _static_activation_hook(bits: int, log_scale: torch.Tensor):
    def hook(_module, inputs):
        return (fake_quantize_static(inputs[0], bits, log_scale),) + tuple(inputs[1:])

    return hook
