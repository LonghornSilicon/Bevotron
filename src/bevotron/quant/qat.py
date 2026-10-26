"""Quantization-aware training: the same fake quantization as PTQ, applied during training
with a straight-through estimator so gradients flow through the rounding.

Weights are kept in full precision (the optimizer updates those); every forward pass
sees their rounded version. After training, PTQ of the checkpoint lands on (almost) the
same grid, so evaluation uses the ordinary PTQ path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevotron.quant.ptq import SCHEME_BITS, fake_quantize


def fake_quantize_ste(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Forward: rounded values. Backward: identity (straight-through)."""
    return x + (fake_quantize(x, bits, dim) - x).detach()


class QATLinear(nn.Linear):
    """nn.Linear whose forward uses fake-quantized weights (and inputs, if a_bits is set).
    Same parameters and state_dict keys as nn.Linear, so checkpoints load into plain models."""

    w_bits: int | None = None
    a_bits: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_ste(self.weight, self.w_bits, dim=1) if self.w_bits else self.weight
        if self.a_bits:
            x = fake_quantize_ste(x, self.a_bits, dim=-1)
        return F.linear(x, w, self.bias)


def apply_qat(model: nn.Module, scheme: str) -> nn.Module:
    """Swap every Linear for a QATLinear and fake-quantize the KV cache, in place."""
    w_bits, a_bits, kv_bits = SCHEME_BITS[scheme]
    for module in list(model.modules()):
        for child in list(module.children()):
            if type(child) is nn.Linear:
                child.__class__ = QATLinear  # same weights, quantized forward
                child.w_bits, child.a_bits = w_bits, a_bits
        if kv_bits is not None and hasattr(module, "kv_quant"):
            module.kv_quant = lambda t, b=kv_bits: fake_quantize_ste(t, b, dim=-1)
    return model
