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


# Windows scored per forward pass in the full validation pass. Independent of the training
# batch: the loss is a mean over every token either way, and the logits for a 131K-token
# batch at a 50k vocabulary are 26 GB, more than a laptop GPU holds.
EVAL_BATCH_TOKENS = 16_384


def eval_batch_size(context_len: int, batch_tokens: int) -> int:
    """Windows per eval forward pass: the training batch, capped at EVAL_BATCH_TOKENS."""
    return max(1, min(batch_tokens, EVAL_BATCH_TOKENS) // context_len)


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
    from bevotron.quant.ptq import quantize_for_eval
    from bevotron.quant.static import split_scales

    meta = prepare(config)
    val = load_split(config, "val")
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
    model = Bevo(config).to(device)
    weights, scales = split_scales(ckpt["model"])
    model.load_state_dict(weights)
    batch = eval_batch_size(config.model.context_len, config.train.batch_tokens)

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
        quantize_for_eval(model, config, scales, device)
        q_loss = full_val_loss(model, val, config.model.context_len, batch, device)
        result.update(val_loss=q_loss, bpb=bits_per_byte(q_loss, meta.val_tokens, meta.val_bytes))
    result["perplexity"] = perplexity(result["val_loss"])
    return result
