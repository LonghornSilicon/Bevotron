"""Deployed size: weight bytes plus KV-cache bytes at the run's precision and context."""

from __future__ import annotations

from dataclasses import dataclass

from bevotron.config import QUANT_BYTES, RunConfig


@dataclass(frozen=True)
class MemoryReport:
    weight_bytes: int
    kv_cache_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_cache_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1e6


def memory(config: RunConfig) -> MemoryReport:
    """Bytes the chip must hold or stream per generated token.

    Weights count non-embedding parameters only (the token table stays on the
    CPU). KV cache = 2 (K and V) x layers x kv heads x head dim x cached tokens,
    where cached tokens is the context length, or the local window if smaller.
    """
    m = config.model
    w_bytes, _, kv_bytes = QUANT_BYTES[config.quant.scheme]
    cached = m.context_len if m.attention_window is None else min(m.attention_window, m.context_len)
    kv = 2 * m.n_layer * m.n_kv_head * m.head_dim * cached
    return MemoryReport(
        weight_bytes=int(config.non_embedding_params() * w_bytes),
        kv_cache_bytes=int(kv * kv_bytes),
    )
