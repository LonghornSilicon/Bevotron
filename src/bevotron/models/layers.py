"""Normalization and position encoding, selected by config."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"unknown norm {kind!r}")


class LearnedPositions(nn.Module):
    """One learned vector per position, added to the token embeddings (GPT-2 style)."""

    def __init__(self, context_len: int, dim: int):
        super().__init__()
        self.table = nn.Embedding(context_len, dim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(offset, offset + x.size(1), device=x.device)
        return x + self.table(positions)


class RotaryEmbedding(nn.Module):
    """RoPE: rotate query/key pairs by a position-dependent angle. No parameters."""

    def __init__(self, head_dim: int, context_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(context_len).float(), inv_freq)  # (T, hd/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """q, k: (B, heads, T, head_dim)."""
        t = q.size(2)
        cos = self.cos[offset : offset + t].to(q.dtype)[None, None]  # (1,1,T,hd/2)
        sin = self.sin[offset : offset + t].to(q.dtype)[None, None]
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
