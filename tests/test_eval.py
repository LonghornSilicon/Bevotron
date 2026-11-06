import math
from pathlib import Path

from bevotron.config import RunConfig, apply_overrides, load_raw
from bevotron.eval.memory import memory
from bevotron.eval.quality import bits_per_byte, eval_batch_size, perplexity

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


def test_embedding_memory_is_separate_and_scales_with_vocab():
    base = load_raw(BASELINE)
    big = memory(RunConfig.from_dict(base))  # int8, 50k vocab, tied, learned positions
    small = memory(RunConfig.from_dict(apply_overrides(base, {"tokenizer.vocab_size": 16_000})))
    positions_fp32 = 512 * 512 * 4  # PTQ never rounds learned positions
    assert big.embedding_bytes == 50_000 * 512 * 1 + positions_fp32
    assert big.total_bytes == small.total_bytes  # memory_mb ignores the vocabulary
    assert big.embedding_bytes - small.embedding_bytes == (50_000 - 16_000) * 512
    assert math.isclose(big.with_embedding_mb, big.total_mb + big.embedding_mb)
    w4 = memory(RunConfig.from_dict(apply_overrides(base, {"quant.scheme": "w4a4kv4"})))
    assert w4.embedding_bytes == 50_000 * 512 // 2 + positions_fp32


def test_untied_input_table_and_positions_count_at_fp32():
    base = load_raw(BASELINE)
    untied = memory(RunConfig.from_dict(apply_overrides(base, {"model.tie_embeddings": False})))
    # output layer rounded to int8; input table and positions left at fp32 by PTQ
    assert untied.embedding_bytes == 50_000 * 512 * (1 + 4) + 512 * 512 * 4
    rope = memory(RunConfig.from_dict(apply_overrides(base, {"model.pos_encoding": "rope"})))
    assert rope.embedding_bytes == 50_000 * 512


def test_ptq_rounds_only_the_tables_memory_counts_as_rounded():
    import torch

    from bevotron.models.transformer import Bevo
    from bevotron.quant.ptq import apply_ptq, fake_quantize

    raw = apply_overrides(load_raw(BASELINE), {"tokenizer.vocab_size": 300, "model.n_layer": 1})
    for tied in (True, False):
        cfg = RunConfig.from_dict(apply_overrides(raw, {"model.tie_embeddings": tied}))
        model = Bevo(cfg)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        apply_ptq(model, "int8")
        head = model.lm_head.weight
        assert torch.equal(
            head, fake_quantize(before["lm_head.weight" if not tied else "wte.weight"], 8, dim=1)
        )
        assert torch.equal(model.wte.weight, head) == tied
        pos = [n for n in before if n.startswith("positions")]
        assert pos and all(torch.equal(before[n], dict(model.named_parameters())[n]) for n in pos)


def test_bits_per_byte_worked_example():
    # "The cat sat." is 12 bytes; model A: 4 tokens at 2.0 nats, model B: 8 tokens at 1.2 nats
    a = bits_per_byte(2.0, 4, 12)
    b = bits_per_byte(1.2, 8, 12)
    assert round(a, 2) == 0.96 and round(b, 2) == 1.15 and a < b
    assert math.isclose(perplexity(math.log(24)), 24)


def test_eval_batch_is_capped_for_large_training_batches():
    # 131K-token training batch at context 512: 256 windows would need 26 GB of logits at a
    # 50k vocabulary; eval scores 32 windows at a time instead. Small batches are unchanged.
    assert eval_batch_size(512, 131_072) == 32
    assert eval_batch_size(512, 16_384) == 32
    assert eval_batch_size(128, 4_096) == 32
    assert eval_batch_size(512, 256) == 1
