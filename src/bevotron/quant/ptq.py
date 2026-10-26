"""Simulated post-training quantization.

Values are rounded to an integer grid and scaled back to float ("fake quant"), so the
model runs in float but only ever sees numbers an int8/int4 chip could represent.
That is enough to measure the accuracy cost; real integer kernels live in the
inference engine.

  weights      symmetric, one scale per output channel, round-to-nearest
  activations  symmetric, one scale per token, applied to every Linear input
  KV cache     symmetric, one scale per token per head, applied as K and V are produced
"""

from __future__ import annotations

import torch
import torch.nn as nn

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


def apply_ptq(model: nn.Module, scheme: str) -> nn.Module:
    """Quantize a trained model in place according to `scheme`."""
    w_bits, a_bits, kv_bits = SCHEME_BITS[scheme]
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight.dim() == 2:
            if w_bits is not None:
                with torch.no_grad():
                    module.weight.copy_(fake_quantize(module.weight, w_bits, dim=1))
            if a_bits is not None:
                module.register_forward_pre_hook(_activation_hook(a_bits))
        if kv_bits is not None and hasattr(module, "kv_quant"):
            module.kv_quant = lambda t, b=kv_bits: fake_quantize(
                t, b, dim=-2
            )  # per token, per head
    return model


def _activation_hook(bits: int):
    def hook(_module, inputs):
        x = inputs[0]
        return (fake_quantize(x, bits, dim=-1),) + tuple(inputs[1:])

    return hook
