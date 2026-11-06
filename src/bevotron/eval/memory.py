"""Deployed size: weight bytes plus KV-cache bytes at the run's precision and context."""

from __future__ import annotations

from dataclasses import dataclass

from bevotron.config import QUANT_BYTES, RunConfig


@dataclass(frozen=True)
class MemoryReport:
    weight_bytes: int
    kv_cache_bytes: int
    embedding_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_cache_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1e6

    @property
    def embedding_mb(self) -> float:
        return self.embedding_bytes / 1e6

    @property
    def with_embedding_mb(self) -> float:
        return (self.total_bytes + self.embedding_bytes) / 1e6


def memory(config: RunConfig) -> MemoryReport:
    """Bytes the chip must hold or stream per generated token.

    Weights count non-embedding parameters only (the token table stays on the
    CPU). KV cache = 2 (K and V) x layers x kv heads x head dim x cached tokens,
    where cached tokens is the context length, or the local window if smaller.

    embedding_bytes is the token table (plus learned positions) at the weight precision,
    kept separate because it is the only part that changes with vocabulary size. With
    tied embeddings the output layer reads the whole table for every generated token, so
    a chip that runs the output layer must hold or stream it as well: with_embedding_mb.
    """
    m = config.model
    w_bytes, _, kv_bytes = QUANT_BYTES[config.quant.scheme]
    cached = m.context_len if m.attention_window is None else min(m.attention_window, m.context_len)
    kv = 2 * m.n_layer * m.n_kv_head * m.head_dim * cached
    return MemoryReport(
        weight_bytes=int(config.non_embedding_params() * w_bytes),
        kv_cache_bytes=int(kv * kv_bytes),
        embedding_bytes=int(config.embedding_params() * w_bytes),
    )
