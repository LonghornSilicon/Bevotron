"""Random fixed-length windows from a token array (nanoGPT's 'poor man's data loader')."""

from __future__ import annotations

import numpy as np
import torch


def get_batch(
    data: np.memmap, batch_size: int, context_len: int, rng: np.random.Generator, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(data) - context_len - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[s : s + context_len].astype(np.int64)) for s in starts])
    y = torch.stack(
        [torch.from_numpy(data[s + 1 : s + 1 + context_len].astype(np.int64)) for s in starts]
    )
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)
