"""Causal self-attention: MHA / GQA / MQA via n_kv_head, optional local window, KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevotron.config import ModelConfig
from bevotron.models.layers import RotaryEmbedding


def causal_mask(t_q: int, t_k: int, window: int | None, device) -> torch.Tensor:
    """(t_q, t_k) boolean mask, True = may attend. Query i sits at key position
    (t_k - t_q + i), so this also works when a KV cache holds earlier tokens."""
    q_pos = torch.arange(t_k - t_q, t_k, device=device)[:, None]
    k_pos = torch.arange(t_k, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed &= q_pos - k_pos < window
    return allowed


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding | None):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.window = cfg.attention_window
        kv_dim = cfg.n_kv_head * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = rope
        self.kv_quant: callable | None = None  # set by quant.ptq to fake-quantize the cache

    def forward(self, x: torch.Tensor, cache: tuple | None = None):
        bsz, t, _ = x.shape
        q = self.q_proj(x).view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        offset = 0 if cache is None else cache[0].size(2)
        if self.rope is not None:
            q, k = self.rope(q, k, offset)
        if self.kv_quant is not None:
            k, v = self.kv_quant(k), self.kv_quant(v)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v)
        if self.n_kv_head != self.n_head:  # GQA / MQA: each KV head serves several Q heads
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        mask = causal_mask(t, k.size(2), self.window, x.device)[None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(bsz, t, self.n_head * self.head_dim)
        return self.o_proj(y), new_cache
