"""Deployed size: weight bytes plus KV-cache bytes at the run's precision and context."""

from __future__ import annotations

from dataclasses import dataclass

from bevotron.config import QUANT_BYTES, RunConfig

# PTQ and QAT round only nn.Linear weights, so tables outside one stay at fp32.
UNROUNDED_BYTES = 4.0


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
    where cached tokens is the context length, or the local window if smaller, per layer
    (global layers keep the whole context).

    embedding_bytes is the embeddings at the precision the evaluated model stores them,
    kept separate because they are the only part that changes with vocabulary size. The
    output layer is an nn.Linear, so its vocab x d_model table is at the weight precision;
    with tied embeddings that is also the input table. Learned positions and an untied
    input table are never rounded, so they count at UNROUNDED_BYTES. The output layer
    reads its whole table for every generated token, so a chip that runs it must hold or
    stream the embeddings as well: with_embedding_mb.
    """
    m = config.model
    w_bytes, _, kv_bytes = QUANT_BYTES[config.quant.scheme]
    kv = 2 * m.n_kv_head * m.head_dim * m.kv_tokens()
    table = config.tokenizer.vocab_size * m.d_model
    positions = m.context_len * m.d_model if m.pos_encoding == "learned" else 0
    unrounded = positions + (0 if m.tie_embeddings else table)
    return MemoryReport(
        weight_bytes=int(config.non_embedding_params() * w_bytes),
        kv_cache_bytes=int(kv * kv_bytes),
        embedding_bytes=int(table * w_bytes + unrounded * UNROUNDED_BYTES),
    )
