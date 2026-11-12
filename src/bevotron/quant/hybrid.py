"""Sampled scales, a mix of dynamic and static: each quantized matrix takes its scale from
a small part of itself, times a fixed headroom factor.

A matrix is what one call produces: a Linear's input (B, T, C) with one row per token, or
one head's keys or values (B, H, T, head_dim). Modes (quant.scales):

  first_row     the max of row 0 sets the scale for every row of the matrix
  first_n       the max of the matrix's first N values (row 0's first N, quant.scale_n)
                sets the scale for every row
  row_first_n   each row's scale comes from that row's own first N values
  first_tile    the max of the top-left N x N tile (rows 0..N-1, values 0..N-1) sets
                the scale for every row

On a chip, first_row keeps one row at int16 to find its max, quantizes it, then writes
every later row straight to int4 with that scale. first_n keeps only N values. row_first_n
keeps N values of every row. first_tile keeps N x N values, and rows N and later go
straight to int4. The scale is one integer per matrix (or per row), so the
consumer's requantization is the static (multiplier, shift) times that integer
(quant.int_engine).

The sampled max can miss larger values later in the matrix (those saturate), so it is
multiplied by a headroom factor, quant.scale_margin. That factor is the static part: one
number per Linear input and per KV head, learned in QAT with the same LSQ gradient as
static scales, and stored as a natural log:

  <linear>.act_log_margin            scalar
  <attention>.kv_quant.k_log_margin  (1, kv_heads, 1, 1)
  <attention>.kv_quant.v_log_margin  (1, kv_heads, 1, 1)

Each call samples its own matrix. In training and evaluation a call is a whole sequence,
so first_row means token 0. In cached generation every new token is a one-row matrix and
gets its own scale (for first_tile, from its first N values); cached keys and values keep
the scale they were written with.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bevotron.quant.static import fake_quantize_static, qmax

MODES = ("first_row", "first_n", "row_first_n", "first_tile")


def sample(x: torch.Tensor, mode: str, n: int) -> torch.Tensor:
    """The part of x (..., T, C) whose max sets the scale; rows are tokens."""
    if mode == "first_row":
        return x[..., :1, :]
    if mode == "first_n":
        return x[..., :1, :n]
    if mode == "row_first_n":
        return x[..., :n]
    if mode == "first_tile":
        return x[..., :n, :n]
    raise ValueError(f"unknown sampled-scale mode {mode!r}; choose {MODES}")


def sample_max(x: torch.Tensor, mode: str, n: int) -> torch.Tensor:
    """max |value| of the sample, shaped to broadcast over x: (..., T, 1) for row_first_n,
    (..., 1, 1) for the modes that set one scale per matrix. Works on integers too."""
    m = sample(x, mode, n).abs().amax(dim=-1, keepdim=True)
    return m if mode == "row_first_n" else m.amax(dim=-2, keepdim=True)


def fake_quantize_sampled(
    x: torch.Tensor, bits: int, mode: str, n: int, log_margin: torch.Tensor
) -> torch.Tensor:
    """Scale = max|sample| * exp(log_margin) / qmax, broadcast over the matrix (first_row,
    first_n, first_tile) or kept per row (row_first_n). Straight-through for x, LSQ for the
    margin; the sampled max itself is not differentiated (as with dynamic scales)."""
    amax = sample_max(x.detach(), mode, n).clamp(min=1e-8)
    return fake_quantize_static(x, bits, (amax / qmax(bits)).log() + log_margin)


class SampledKV(nn.Module):
    """Sampled scales for the cached keys and values, one learned margin per head."""

    def __init__(self, bits: int, mode: str, n: int, k_log_margin, v_log_margin):
        super().__init__()
        self.bits, self.mode, self.n = bits, mode, n
        self.k_log_margin = nn.Parameter(k_log_margin.detach().clone())
        self.v_log_margin = nn.Parameter(v_log_margin.detach().clone())

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            fake_quantize_sampled(k, self.bits, self.mode, self.n, self.k_log_margin),
            fake_quantize_sampled(v, self.bits, self.mode, self.n, self.v_log_margin),
        )


def start_margins(model: nn.Module, margin: float, a_bits, kv_bits) -> dict:
    """Log margins for every Linear input and KV head, all set to `margin` (QAT learns
    them from here; PTQ uses them as they are). Read before quantizing the model."""
    log = math.log(margin)
    out = {}
    for name, mod in model.named_modules():
        if a_bits is not None and isinstance(mod, nn.Linear):
            out[f"{name}.act_log_margin"] = torch.tensor(log)
        if kv_bits is not None and hasattr(mod, "kv_quant"):
            for which in ("k", "v"):
                out[f"{name}.kv_quant.{which}_log_margin"] = torch.full(
                    (1, mod.n_kv_head, 1, 1), log
                )
    return out


def sampling_of(quant_config) -> tuple[str, int] | None:
    """(mode, n) for a sampled-scale run, None for dynamic or static."""
    if quant_config.scales not in MODES:
        return None
    return quant_config.scales, quant_config.scale_n
