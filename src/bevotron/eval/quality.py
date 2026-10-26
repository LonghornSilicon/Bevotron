"""Validation quality: perplexity (primary), bits per byte, and a hook for the LLM judge.

Perplexity = exp(mean cross-entropy per token) on the validation split. Bits per byte is
the same loss normalized by bytes of text instead of tokens; report it alongside perplexity
whenever runs use different tokenizers (vocab study), since per-token numbers are not
comparable across tokenizers.
"""

from __future__ import annotations

import math
from pathlib import Path

from bevotron.config import RunConfig


def bits_per_byte(val_loss_nats: float, val_tokens: int, val_bytes: int) -> float:
    if val_tokens <= 0 or val_bytes <= 0:
        raise ValueError("val_tokens and val_bytes must be positive")
    return val_loss_nats / math.log(2) * val_tokens / val_bytes


def perplexity(val_loss_nats: float) -> float:
    """Same information as the loss, exponentiated; for intuition only."""
    return math.exp(val_loss_nats)


def full_val_loss(model, data, context_len: int, batch_size: int, device: str) -> float:
    """Mean cross-entropy over the whole validation array, in non-overlapping windows."""
    import numpy as np
    import torch

    model.eval()
    n_windows = (len(data) - 1) // context_len
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            idx = range(start, min(start + batch_size, n_windows))
            x = torch.stack(
                [
                    torch.from_numpy(data[i * context_len : (i + 1) * context_len].astype(np.int64))
                    for i in idx
                ]
            ).to(device)
            y = torch.stack(
                [
                    torch.from_numpy(
                        data[i * context_len + 1 : (i + 1) * context_len + 1].astype(np.int64)
                    )
                    for i in idx
                ]
            ).to(device)
            _, loss, _ = model(x, y)
            total += loss.item() * y.numel()
            count += y.numel()
    model.train()
    return total / count


def evaluate_run(config: RunConfig, run_dir: Path, device: str = "cpu") -> dict:
    """Score a checkpoint at full precision and under the run's quantization scheme."""
    import torch

    from bevotron.data.prepare import load_split, prepare
    from bevotron.eval.memory import memory
    from bevotron.models import Bevo
    from bevotron.quant.ptq import apply_ptq

    meta = prepare(config)
    val = load_split(config, "val")
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
    model = Bevo(config).to(device)
    model.load_state_dict(ckpt["model"])
    batch = max(1, config.train.batch_tokens // config.model.context_len)

    fp_loss = full_val_loss(model, val, config.model.context_len, batch, device)
    result = {
        "scheme": config.quant.scheme,
        "mode": config.quant.mode,
        "val_loss_fp": fp_loss,
        "perplexity_fp": perplexity(fp_loss),
        "bpb_fp": bits_per_byte(fp_loss, meta.val_tokens, meta.val_bytes),
        "memory_mb": memory(config).total_mb,
        "checkpoint_step": ckpt["step"],
    }
    if config.quant.scheme == "none":
        result.update(val_loss=fp_loss, bpb=result["bpb_fp"])
    else:
        apply_ptq(model, config.quant.scheme)
        q_loss = full_val_loss(model, val, config.model.context_len, batch, device)
        result.update(val_loss=q_loss, bpb=bits_per_byte(q_loss, meta.val_tokens, meta.val_bytes))
    result["perplexity"] = perplexity(result["val_loss"])
    return result
