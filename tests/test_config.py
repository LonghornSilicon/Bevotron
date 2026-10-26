from pathlib import Path

import pytest

from bevotron.config import RunConfig, apply_overrides, load_raw, parse_count

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def test_baseline_loads_and_matches_known_counts():
    cfg = RunConfig.load(BASELINE)
    assert cfg.model.norm == "layernorm" and cfg.model.pos_encoding == "learned"
    assert cfg.params_per_layer() == 12 * 512**2  # 4d^2 attention + 8d^2 MLP
    assert cfg.non_embedding_params() == 25_165_824
    assert cfg.total_params() == 25_165_824 + 50_000 * 512 + 512 * 512
    assert cfg.train.total_tokens == 504_000_000 and cfg.train.steps == 30_761


def test_parse_count():
    assert parse_count("504M") == 504_000_000
    assert parse_count("2.1B") == 2_100_000_000
    assert parse_count("16K") == 16_000
    assert parse_count(123) == 123


def test_run_id_is_stable_and_changes_with_config():
    a = RunConfig.load(BASELINE)
    b = RunConfig.load(BASELINE)
    assert a.run_id() == b.run_id()
    c = RunConfig.from_dict(apply_overrides(load_raw(BASELINE), {"model.n_layer": 4}))
    assert c.run_id() != a.run_id()


def test_gqa_and_swiglu_param_counts():
    base = load_raw(BASELINE)
    gqa = RunConfig.from_dict(apply_overrides(base, {"model.n_kv_head": 4}))
    assert gqa.params_per_layer() == 2 * 512**2 + 2 * 512 * 128 + 8 * 512**2
    llama = RunConfig.from_dict(apply_overrides(base, {"model.preset": "llama"}))
    assert llama.model.mlp_act == "swiglu" and llama.embedding_params() == 50_000 * 512
    assert llama.params_per_layer() == 4 * 512**2 + 3 * 512 * 2048


@pytest.mark.parametrize(
    "override",
    [
        {"model.n_head": 15},  # d_model not divisible
        {"model.n_kv_head": 3},  # n_head not divisible
        {"model.preset": "gpt_neo"},  # unknown preset
        {"quant.scheme": "fp8"},
        {"train.tokens": "100"},  # less than one batch
        {"tokenizer.vocab_size": 100},
    ],
)
def test_invalid_configs_are_rejected(override):
    base = load_raw(BASELINE)
    with pytest.raises(ValueError):
        RunConfig.from_dict(apply_overrides(base, override))


def test_unknown_keys_are_rejected():
    base = load_raw(BASELINE)
    with pytest.raises(ValueError):
        RunConfig.from_dict(apply_overrides(base, {"model.n_layers": 4}))  # typo


def test_save_round_trip(tmp_path):
    cfg = RunConfig.load(BASELINE)
    cfg.save(tmp_path / "c.yaml")
    assert RunConfig.load(tmp_path / "c.yaml").run_id() == cfg.run_id()


def test_training_key_shared_by_ptq_schemes_but_not_qat():
    base = load_raw(BASELINE)
    a = RunConfig.from_dict(apply_overrides(base, {"quant.scheme": "int8"}))
    b = RunConfig.from_dict(apply_overrides(base, {"quant.scheme": "w4a4"}))
    c = RunConfig.from_dict(apply_overrides(base, {"quant.scheme": "w4a4", "quant.mode": "qat"}))
    d = RunConfig.from_dict(apply_overrides(base, {"quant.scheme": "w4a4", "model.n_layer": 4}))
    assert a.run_id() != b.run_id() and a.training_key() == b.training_key()
    assert c.training_key() != a.training_key()
    assert d.training_key() != a.training_key()
