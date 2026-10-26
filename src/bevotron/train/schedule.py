"""Linear warmup, then cosine decay to min_lr_ratio x lr (nanoGPT / Chinchilla style)."""

from __future__ import annotations

import math


def lr_at(step: int, total_steps: int, lr: float, warmup_frac: float, min_lr_ratio: float) -> float:
    warmup = max(1, int(warmup_frac * total_steps))
    min_lr = lr * min_lr_ratio
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)
