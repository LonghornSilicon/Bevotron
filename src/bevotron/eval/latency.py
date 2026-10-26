"""Latency: a bandwidth lower bound until the architecture team's performance model exists.

Reef streams weights and KV cache from the host per generated token, so the floor on
per-token latency is bytes moved / link bandwidth. Compute is ignored here.
"""

from __future__ import annotations

from bevotron.config import RunConfig
from bevotron.eval.memory import memory

DEFAULT_LINK_MB_PER_S = 150.0  # CPU-to-NPU link, from the architecture team's estimate


def latency_floor_ms(config: RunConfig, link_mb_per_s: float = DEFAULT_LINK_MB_PER_S) -> float:
    """Milliseconds per generated token if the link were the only cost."""
    return memory(config).total_bytes / (link_mb_per_s * 1e6) * 1e3
