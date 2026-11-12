"""Static quantization scales: one fixed scale per Linear input and per KV-cache head,
instead of a fresh scale for every token.

With static scales an integer chip folds each layer's scales into one integer multiplier
and shift, so requantization needs no float math and no per-token max. Scales start at a
percentile of calibration activations (clipping rare outliers), and QAT then learns them:
gradients reach the scale through the rounding, as in LSQ.

Scales are stored as natural logs under these state_dict suffixes:
  <linear>.act_log_scale            scalar, the Linear's input
  <attention>.kv_quant.k_log_scale  (1, kv_heads, 1, 1), the cached keys
  <attention>.kv_quant.v_log_scale  (1, kv_heads, 1, 1), the cached values
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# Learned quantization parameters: static log scales, and the log headroom factors of
# sampled scales (quant.hybrid). Split from the weights on load, own optimizer group.
SCALE_SUFFIXES = (
    "act_log_scale", "k_log_scale", "v_log_scale",
    "act_log_margin", "k_log_margin", "v_log_margin",
)  # fmt: skip
CALIB_PERCENTILE = 0.9999  # clip the top 0.01% of values; the max wastes the grid on outliers
CALIB_BATCHES = 8
CALIB_WINDOWS = 16  # windows per calibration batch
CALIB_SEED = 0
SAMPLES_PER_BATCH = 200_000  # values kept per tensor per batch for the percentile
SCALE_LR_MULT = 10.0  # learned log scales move faster than weights (lr x 10, no decay)


def qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def fake_quantize_static(x: torch.Tensor, bits: int, log_scale: torch.Tensor) -> torch.Tensor:
    """Round x to a fixed grid of step exp(log_scale). Straight-through for x; the scale
    gets the LSQ gradient (rounding error inside the range, the clip value outside)."""
    s = log_scale.exp()
    y = x / s
    y = y + (torch.round(y) - y).detach()
    return torch.clamp(y, -qmax(bits) - 1, qmax(bits)) * s


class StaticKV(nn.Module):
    """Fixed per-head scales for the cached keys and values."""

    def __init__(self, bits: int, k_log_scale: torch.Tensor, v_log_scale: torch.Tensor):
        super().__init__()
        self.bits = bits
        self.k_log_scale = nn.Parameter(k_log_scale.detach().clone())
        self.v_log_scale = nn.Parameter(v_log_scale.detach().clone())

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            fake_quantize_static(k, self.bits, self.k_log_scale),
            fake_quantize_static(v, self.bits, self.v_log_scale),
        )


def split_scales(state: dict) -> tuple[dict, dict]:
    """(plain model weights, learned scales or margins) from a checkpoint's state_dict."""
    scales = {k: v for k, v in state.items() if k.endswith(SCALE_SUFFIXES)}
    return {k: v for k, v in state.items() if k not in scales}, scales


def _sample(values: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    flat = values.detach().abs().flatten().float()
    if flat.numel() > SAMPLES_PER_BATCH:
        idx = torch.randint(0, flat.numel(), (SAMPLES_PER_BATCH,), generator=gen)
        flat = flat[idx.to(flat.device)]
    return flat.cpu()


@torch.no_grad()
def calibrate(model: nn.Module, batches, a_bits: int | None, kv_bits: int | None) -> dict:
    """Run a plain (unquantized) model on `batches` and return starting log scales."""
    gen = torch.Generator().manual_seed(CALIB_SEED)
    acts: dict[str, list] = defaultdict(list)
    kv: dict[str, list] = defaultdict(list)
    handles, attentions = [], []
    for name, mod in model.named_modules():
        if a_bits is not None and isinstance(mod, nn.Linear):

            def hook(_m, inputs, key=f"{name}.act_log_scale"):
                acts[key].append(_sample(inputs[0], gen))

            handles.append(mod.register_forward_pre_hook(hook))
        if kv_bits is not None and hasattr(mod, "kv_quant"):
            if mod.kv_quant is not None:
                raise ValueError("calibrate a plain model, before quantizing it")

            def record(k, v, key=name):
                for which, t in (("k", k), ("v", v)):
                    per_head = t.transpose(0, 1).reshape(t.size(1), -1)  # (heads, values)
                    kv[f"{key}.kv_quant.{which}_log_scale"].append(
                        torch.stack([_sample(h, gen) for h in per_head])
                    )
                return k, v

            mod.kv_quant = record
            attentions.append(mod)
    was_training = model.training
    model.eval()
    try:
        for x in batches:
            model(x)
    finally:
        for h in handles:
            h.remove()
        for mod in attentions:
            mod.kv_quant = None
        model.train(was_training)

    scales = {}
    for key, chunks in acts.items():
        top = torch.quantile(torch.cat(chunks), CALIB_PERCENTILE).clamp(min=1e-8)
        scales[key] = (top / qmax(a_bits)).log()
    for key, chunks in kv.items():
        top = torch.quantile(torch.cat(chunks, dim=1), CALIB_PERCENTILE, dim=1).clamp(min=1e-8)
        scales[key] = (top / qmax(kv_bits)).log().view(1, -1, 1, 1)
    return scales


def calibration_batches(config, device: str) -> list[torch.Tensor]:
    """Fixed training windows (never validation) for the starting scales."""
    from bevotron.data.loader import get_batch
    from bevotron.data.prepare import load_split

    data = load_split(config, "train")
    rng = np.random.default_rng(CALIB_SEED)
    ctx = config.model.context_len
    return [get_batch(data, CALIB_WINDOWS, ctx, rng, device)[0] for _ in range(CALIB_BATCHES)]
