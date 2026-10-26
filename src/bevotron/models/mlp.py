"""Feed-forward block: GELU (2 matrices) or SwiGLU (3 matrices), width = mlp_ratio x d_model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevotron.config import ModelConfig


class GeluMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.d_model)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.d_model)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def make_mlp(cfg: ModelConfig) -> nn.Module:
    if cfg.mlp_act == "gelu":
        return GeluMLP(cfg)
    if cfg.mlp_act == "swiglu":
        return SwiGLU(cfg)
    raise ValueError(f"unknown mlp_act {cfg.mlp_act!r}")
