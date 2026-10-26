"""Model and PTQ tests. Need torch; skipped where it is not installed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bevotron.config import RunConfig, apply_overrides, load_raw  # noqa: E402
from bevotron.models import Bevo  # noqa: E402
from bevotron.models.attention import causal_mask  # noqa: E402
from bevotron.quant.ptq import apply_ptq, fake_quantize  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def tiny(**overrides):
    base = {
        "model.n_layer": 2,
        "model.d_model": 64,
        "model.n_head": 4,
        "model.n_kv_head": 4,
        "model.context_len": 32,
        "tokenizer.vocab_size": 300,
        "train.tokens": "32K",
        "train.batch_tokens": 128,
    }
    return RunConfig.from_dict(apply_overrides(load_raw(BASELINE), {**base, **overrides}))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"model.n_kv_head": 1},
        {"model.preset": "llama"},
        {"model.attention_window": 8},
        {"model.mlp_ratio": 2},
    ],
)
def test_param_count_matches_config_formula(overrides):
    cfg = tiny(**overrides)
    model = Bevo(cfg)
    assert model.non_embedding_params() == cfg.non_embedding_params()


def test_forward_shapes_and_loss_near_uniform_at_init():
    cfg = tiny()
    model = Bevo(cfg)
    x = torch.randint(0, cfg.tokenizer.vocab_size, (3, cfg.model.context_len))
    logits, loss, caches = model(x, x)
    assert logits.shape == (3, cfg.model.context_len, cfg.tokenizer.vocab_size)
    assert len(caches) == cfg.model.n_layer
    # tied embeddings inflate the current token's logit at init; the gap shrinks with depth/width
    assert abs(loss.item() - torch.log(torch.tensor(300.0)).item()) < 1.0


def test_causal_mask_and_window():
    m = causal_mask(4, 4, None, "cpu")
    assert m.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]
    w = causal_mask(4, 4, 2, "cpu")
    assert w[3].tolist() == [False, False, True, True]
    # with a cache of 3 earlier tokens, 1 new query attends to all 4 keys (full window)
    assert causal_mask(1, 4, None, "cpu").tolist() == [[True, True, True, True]]


def test_kv_cache_matches_full_recompute():
    cfg = tiny(**{"model.n_kv_head": 2})
    model = Bevo(cfg).eval()
    x = torch.randint(0, cfg.tokenizer.vocab_size, (1, 10))
    with torch.no_grad():
        full, _, _ = model(x)
        _, _, caches = model(x[:, :7])
        step, _, _ = model(x[:, 7:], caches=caches)
    assert torch.allclose(full[:, 7:], step, atol=1e-4)


def test_generate_runs():
    cfg = tiny()
    out = Bevo(cfg).eval().generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=5)
    assert out.shape == (1, 6)


def test_fake_quantize_is_idempotent_and_bounded():
    x = torch.randn(16, 32)
    q = fake_quantize(x, 8, dim=1)
    assert torch.allclose(fake_quantize(q, 8, dim=1), q)
    assert (q - x).abs().max() <= x.abs().amax(dim=1).max() / 127 + 1e-6
    coarse = fake_quantize(x, 4, dim=1)
    assert (coarse - x).abs().mean() > (q - x).abs().mean()


def test_ptq_changes_outputs_only_slightly_at_int8():
    cfg = tiny()
    model = Bevo(cfg).eval()
    x = torch.randint(0, cfg.tokenizer.vocab_size, (2, 16))
    with torch.no_grad():
        before, _, _ = model(x)
        apply_ptq(model, "int8")
        after, _, _ = model(x)
    assert not torch.equal(before, after)
    assert (before - after).abs().mean() < 0.05 * before.abs().mean() + 1e-3


def test_qat_keeps_state_dict_keys_and_passes_gradients():
    from bevotron.quant.qat import apply_qat

    cfg = tiny()
    plain_keys = set(Bevo(cfg).state_dict())
    model = apply_qat(Bevo(cfg), "w4a8kv4")
    assert set(model.state_dict()) == plain_keys  # checkpoints load into a plain model
    x = torch.randint(0, cfg.tokenizer.vocab_size, (2, 16))
    _, loss, _ = model(x, x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
