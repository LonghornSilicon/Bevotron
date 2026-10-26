import math
from pathlib import Path

from bevotron.config import RunConfig, apply_overrides, load_raw
from bevotron.eval.memory import memory
from bevotron.eval.quality import bits_per_byte, perplexity

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def test_baseline_memory_matches_hand_calculation():
    cfg = RunConfig.load(BASELINE)  # int8: 25.2 MB weights + 4.2 MB KV cache
    rep = memory(cfg)
    assert rep.weight_bytes == 25_165_824
    assert rep.kv_cache_bytes == 2 * 8 * 16 * 32 * 512
    assert round(rep.total_mb, 1) == 29.4


def test_memory_responds_to_knobs():
    base = load_raw(BASELINE)
    mqa = memory(RunConfig.from_dict(apply_overrides(base, {"model.n_kv_head": 1})))
    assert mqa.kv_cache_bytes == 2 * 8 * 1 * 32 * 512
    w4 = memory(RunConfig.from_dict(apply_overrides(base, {"quant.scheme": "w4a8kv4"})))
    assert w4.weight_bytes == 25_165_824 // 2 and w4.kv_cache_bytes == 2 * 8 * 16 * 32 * 512 // 2
    local = memory(RunConfig.from_dict(apply_overrides(base, {"model.attention_window": 256})))
    assert local.kv_cache_bytes == 2 * 8 * 16 * 32 * 256


def test_bits_per_byte_worked_example():
    # "The cat sat." is 12 bytes; model A: 4 tokens at 2.0 nats, model B: 8 tokens at 1.2 nats
    a = bits_per_byte(2.0, 4, 12)
    b = bits_per_byte(1.2, 8, 12)
    assert round(a, 2) == 0.96 and round(b, 2) == 1.15 and a < b
    assert math.isclose(perplexity(math.log(24)), 24)
