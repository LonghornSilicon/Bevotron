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
        {"model.attention_window": 8, "model.global_layers": "first_last"},
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


def test_kv_cache_quantization_scales_per_token():
    # K and V are (B, heads, T, head_dim): every token of every head gets its own scale,
    # so one loud token cannot wash out the rest of the cache (per-channel across tokens would).
    heads, t, head_dim = 4, 8, 16
    x = torch.randn(1, heads, t, head_dim)
    x[0, :, 0, :] *= 50
    q = fake_quantize(x, 4, dim=-1)
    scales = q.abs().amax(dim=-1) / 7  # (1, heads, T): grid step each token was rounded to
    assert torch.allclose(scales, x.abs().amax(dim=-1) / 7)
    assert (scales[0, :, 1:].std(dim=-1) > 0).all()  # differ across tokens
    assert (scales[0, :, 0] > 5 * scales[0, :, 1:].amax(dim=-1)).all()
    model = apply_ptq(Bevo(tiny()), "w4a4kv4")
    kq, vq = model.blocks[0].attn.kv_quant(x, x)
    assert torch.equal(kq, q) and torch.equal(vq, q)


@pytest.mark.parametrize(
    "global_layers,windows",
    [
        ("none", [16, 16, 16, 16]),
        ("first", [None, 16, 16, 16]),
        ("last", [16, 16, 16, None]),
        ("first_last", [None, 16, 16, None]),
    ],
)
def test_global_layers_pick_which_layers_see_the_whole_context(global_layers, windows):
    cfg = tiny(
        **{"model.n_layer": 4, "model.attention_window": 16, "model.global_layers": global_layers}
    )
    assert [cfg.model.layer_window(i) for i in range(4)] == windows
    assert [b.attn.window for b in Bevo(cfg).blocks] == windows
    assert cfg.model.kv_tokens() == sum(32 if w is None else w for w in windows)


def test_global_layers_config_rules_and_run_ids():
    raw = load_raw(BASELINE)
    assert "global_layers" not in RunConfig.from_dict(raw).to_dict()["model"]  # IDs kept
    with pytest.raises(ValueError, match="needs an attention_window"):
        RunConfig.from_dict(apply_overrides(raw, {"model.global_layers": "first"}))
    with pytest.raises(ValueError, match="global_layers must be"):
        RunConfig.from_dict(
            apply_overrides(raw, {"model.attention_window": 16, "model.global_layers": "middle"})
        )
    ids = {
        RunConfig.from_dict(
            apply_overrides(raw, {"model.attention_window": 16, "model.global_layers": g})
        ).run_id()
        for g in ("none", "first", "last", "first_last")
    }
    assert len(ids) == 4


@pytest.mark.parametrize(
    "global_layers,reaches", [("none", False), ("first", True), ("last", True)]
)
def test_only_a_global_layer_lets_the_last_token_see_the_first(global_layers, reaches):
    # 2 layers with window 4 see at most 2 * 3 tokens back; token 0 is 31 back.
    cfg = tiny(**{"model.attention_window": 4, "model.global_layers": global_layers})
    torch.manual_seed(0)
    model = Bevo(cfg).eval()
    x = torch.randint(0, cfg.tokenizer.vocab_size, (1, 32))
    y = x.clone()
    y[0, 0] = (x[0, 0] + 1) % cfg.tokenizer.vocab_size
    with torch.no_grad():
        a, _, _ = model(x)
        b, _, _ = model(y)
    assert torch.equal(a[0, -1], b[0, -1]) is not reaches
