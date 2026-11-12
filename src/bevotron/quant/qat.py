"""Quantization-aware training: the same fake quantization as PTQ, applied during training
with a straight-through estimator so gradients flow through the rounding.

Weights are kept in full precision (the optimizer updates those); every forward pass
sees their rounded version. After training, PTQ of the checkpoint lands on (almost) the
same grid, so evaluation uses the ordinary PTQ path.

With static scales (quant.scales: static) each Linear input and each KV-cache head gets a
fixed scale, started from calibration and learned alongside the weights. Those scales are
extra state_dict entries (see quant.static); dynamic QAT adds none.

With sampled scales (quant.scales: first_row, first_n, row_first_n) each matrix takes its
scale from part of itself, and only the headroom factor on top is learned (quant.hybrid).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevotron.quant.hybrid import SampledKV, fake_quantize_sampled
from bevotron.quant.ptq import SCHEME_BITS, fake_quantize
from bevotron.quant.static import StaticKV, fake_quantize_static


def fake_quantize_ste(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Forward: rounded values. Backward: identity (straight-through)."""
    return x + (fake_quantize(x, bits, dim) - x).detach()


class QATLinear(nn.Linear):
    """nn.Linear whose forward uses fake-quantized weights (and inputs, if a_bits is set).
    Dynamic: same parameters and state_dict keys as nn.Linear, so checkpoints load into
    plain models. Static: one more parameter, act_log_scale. Sampled: act_log_margin."""

    w_bits: int | None = None
    a_bits: int | None = None
    sampling: tuple[str, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_ste(self.weight, self.w_bits, dim=1) if self.w_bits else self.weight
        if self.a_bits:
            log_scale = getattr(self, "act_log_scale", None)
            if self.sampling is not None:
                x = fake_quantize_sampled(x, self.a_bits, *self.sampling, self.act_log_margin)
            elif log_scale is None:
                x = fake_quantize_ste(x, self.a_bits, dim=-1)
            else:
                x = fake_quantize_static(x, self.a_bits, log_scale)
        return F.linear(x, w, self.bias)


def apply_qat(
    model: nn.Module, scheme: str, scales: dict | None = None, sampling: tuple | None = None
) -> nn.Module:
    """Swap every Linear for a QATLinear and fake-quantize the KV cache, in place.
    `scales` (log scales from quant.static.calibrate) switches to learned static scales.
    With `sampling` = (mode, n), `scales` holds starting log margins instead
    (quant.hybrid.start_margins) and the scales are sampled from each matrix."""
    w_bits, a_bits, kv_bits = SCHEME_BITS[scheme]
    for name, module in list(model.named_modules()):
        if type(module) is nn.Linear:
            module.__class__ = QATLinear  # same weights, quantized forward
            module.w_bits, module.a_bits = w_bits, a_bits
            if scales is not None and a_bits is not None:
                kind = "margin" if sampling else "scale"
                start = scales[f"{name}.act_log_{kind}"].to(module.weight.device)
                setattr(module, f"act_log_{kind}", nn.Parameter(start.detach().clone()))
                module.sampling = sampling
        if kv_bits is not None and hasattr(module, "kv_quant"):
            if scales is None:
                module.kv_quant = lambda k, v, b=kv_bits: (
                    fake_quantize_ste(k, b, dim=-1),
                    fake_quantize_ste(v, b, dim=-1),
                )
            elif sampling is not None:
                device = next(module.parameters()).device
                module.kv_quant = SampledKV(
                    kv_bits,
                    *sampling,
                    scales[f"{name}.kv_quant.k_log_margin"].to(device),
                    scales[f"{name}.kv_quant.v_log_margin"].to(device),
                )
            else:
                device = next(module.parameters()).device
                module.kv_quant = StaticKV(
                    kv_bits,
                    scales[f"{name}.kv_quant.k_log_scale"].to(device),
                    scales[f"{name}.kv_quant.v_log_scale"].to(device),
                )
    return model
