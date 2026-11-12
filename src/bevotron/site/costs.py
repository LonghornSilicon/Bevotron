"""Hardware cost of one model under its quantization scheme, per generated token.

Reef streams weights and the KV cache from the host for every token (see eval/latency.py),
so decode time is bounded by bytes moved / link bandwidth. If the chip also runs the output
layer, it streams the whole vocab x d_model table as well. The site recomputes ms/token from
these byte counts for whatever link bandwidth the viewer picks.
"""

from __future__ import annotations

from bevotron.config import QUANT_BYTES, RunConfig
from bevotron.eval.latency import DEFAULT_LINK_MB_PER_S
from bevotron.eval.memory import memory


def hardware_cost(config: RunConfig) -> dict:
    m = config.model
    w_bytes, a_bytes, kv_bytes = QUANT_BYTES[config.quant.scheme]
    mem = memory(config)
    output_table = config.tokenizer.vocab_size * m.d_model
    host_bytes = mem.total_bytes
    chip_bytes = host_bytes + output_table * w_bytes
    attention_macs = 2 * m.d_model * m.kv_tokens()  # q.k and attn.v over each layer's context
    return {
        "weight_bits": int(w_bytes * 8),
        "act_bits": int(a_bytes * 8),
        "kv_bits": int(kv_bytes * 8),
        "weights_mb": mem.weight_bytes / 1e6,
        "kv_cache_mb": mem.kv_cache_bytes / 1e6,
        "output_table_mb": output_table * w_bytes / 1e6,
        "output_table_params": output_table,
        "embedding_mb": mem.embedding_mb,
        "memory_mb": mem.total_mb,
        "stored_mb": mem.with_embedding_mb,
        "stream_mb_host_output": host_bytes / 1e6,
        "stream_mb_chip_output": chip_bytes / 1e6,
        "macs_per_token": config.non_embedding_params() + output_table + attention_macs,
        "non_embedding_params": config.non_embedding_params(),
        "embedding_params": config.embedding_params(),
        "total_params": config.total_params(),
        "ms_per_token": chip_bytes / (DEFAULT_LINK_MB_PER_S * 1e6) * 1e3,
    }
