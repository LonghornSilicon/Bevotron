"""bevo: one pre-norm decoder-only transformer, every knob read from RunConfig."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevotron.config import RunConfig
from bevotron.models.attention import CausalSelfAttention
from bevotron.models.layers import LearnedPositions, RotaryEmbedding, make_norm
from bevotron.models.mlp import make_mlp


class Block(nn.Module):
    def __init__(self, config: RunConfig, rope: RotaryEmbedding | None):
        super().__init__()
        m = config.model
        self.norm1 = make_norm(m.norm, m.d_model)
        self.attn = CausalSelfAttention(m, rope)
        self.norm2 = make_norm(m.norm, m.d_model)
        self.mlp = make_mlp(m)

    def forward(self, x: torch.Tensor, cache: tuple | None = None):
        a, cache = self.attn(self.norm1(x), cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, cache


class Bevo(nn.Module):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.config = config
        m, vocab = config.model, config.tokenizer.vocab_size
        self.wte = nn.Embedding(vocab, m.d_model)
        self.positions = (
            LearnedPositions(m.context_len, m.d_model) if m.pos_encoding == "learned" else None
        )
        rope = RotaryEmbedding(m.head_dim, m.context_len) if m.pos_encoding == "rope" else None
        self.blocks = nn.ModuleList(Block(config, rope) for _ in range(m.n_layer))
        self.norm_f = make_norm(m.norm, m.d_model)
        self.lm_head = nn.Linear(m.d_model, vocab, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2 residual scaling
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * m.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def non_embedding_params(self) -> int:
        skip = {id(self.wte.weight), id(self.lm_head.weight)}
        if self.positions is not None:
            skip.add(id(self.positions.table.weight))
        # 2-D weights only, matching RunConfig.non_embedding_params (norm vectors/biases ignored)
        return sum(p.numel() for p in self.parameters() if id(p) not in skip and p.dim() >= 2)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, caches=None):
        """idx: (B, T) token ids. Returns (logits, loss, caches); loss is None without targets."""
        x = self.wte(idx)
        offset = 0 if caches is None else caches[0][0].size(2)
        if self.positions is not None:
            x = self.positions(x, offset)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[i])
            new_caches.append(cache)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        caches = None
        out = idx
        context_len = self.config.model.context_len
        for _ in range(max_new_tokens):
            step_input = out if caches is None else out[:, -1:]
            if out.size(1) > context_len:  # cache would exceed the position table
                caches, step_input = None, out[:, -context_len:]
            logits, _, caches = self(step_input, caches=caches)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            out = torch.cat((out, next_id), dim=1)
        return out

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())
