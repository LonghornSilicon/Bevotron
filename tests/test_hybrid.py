"""Sampled scales (first_row, first_n, row_first_n): config, the fake quant, learned margins
in QAT, and the PTQ path that evaluates them. Need torch; skipped where it is not installed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bevotron.config import RunConfig, apply_overrides, load_raw  # noqa: E402
from bevotron.models import Bevo  # noqa: E402
from bevotron.quant.hybrid import (  # noqa: E402
    ScaleMemory,
    carried_max,
    fake_quantize_sampled,
    sample,
    sample_max,
    sampling_of,
    start_margins,
)
from bevotron.quant.ptq import SCHEME_BITS, apply_ptq, fake_quantize  # noqa: E402
from bevotron.quant.qat import apply_qat  # noqa: E402
from bevotron.quant.static import split_scales  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def tiny(**overrides):
    base = {
        "model.n_layer": 2,
        "model.d_model": 64,
        "model.n_head": 4,
        "model.n_kv_head": 2,
        "model.context_len": 32,
        "tokenizer.vocab_size": 300,
        "train.tokens": "32K",
        "train.batch_tokens": 128,
    }
    return RunConfig.from_dict(apply_overrides(load_raw(BASELINE), {**base, **overrides}))


def test_new_keys_leave_earlier_run_ids_alone():
    raw = load_raw(BASELINE)
    for scales in ("dynamic", "static"):
        cfg = RunConfig.from_dict(apply_overrides(raw, {"quant.scales": scales}))
        quant = cfg.to_dict()["quant"]
        assert "scale_n" not in quant and "scale_margin" not in quant
    first_row = RunConfig.from_dict(apply_overrides(raw, {"quant.scales": "first_row"}))
    assert "scale_n" not in first_row.to_dict()["quant"]  # first_row reads the whole row
    ids = set()
    for n in (8, 128):
        cfg = RunConfig.from_dict(
            apply_overrides(raw, {"quant.scales": "first_n", "quant.scale_n": n})
        )
        assert cfg.to_dict()["quant"]["scale_n"] == n
        assert RunConfig.from_dict(cfg.to_dict()).run_id() == cfg.run_id()
        ids.add(cfg.run_id())
    assert len(ids) == 2
    with pytest.raises(ValueError, match="quant.scales"):
        RunConfig.from_dict(apply_overrides(raw, {"quant.scales": "first_col"}))
    with pytest.raises(ValueError, match="scale_margin"):
        RunConfig.from_dict(
            apply_overrides(raw, {"quant.scales": "first_n", "quant.scale_margin": 0})
        )
    assert sampling_of(first_row.quant) == ("first_row", 128)
    assert sampling_of(RunConfig.from_dict(raw).quant) is None


def test_samples_are_the_parts_named():
    x = torch.arange(2 * 3 * 4 * 6).view(2, 3, 4, 6)  # (B, H, T, C)
    assert torch.equal(sample(x, "first_row", 0), x[:, :, :1, :])
    assert torch.equal(sample(x, "first_n", 2), x[:, :, :1, :2])
    assert torch.equal(sample(x, "row_first_n", 2), x[:, :, :, :2])
    assert torch.equal(sample(x, "first_tile", 2), x[:, :, :2, :2])
    tile_max = sample_max(x, "first_tile", 2)
    assert tile_max.shape == (2, 3, 1, 1)  # one scale per matrix
    assert torch.equal(tile_max, x[:, :, 1:2, 1:2])  # the tile's bottom-right is its max
    assert sample_max(x, "row_first_n", 2).shape == (2, 3, 4, 1)  # one per row


def test_whole_row_sample_is_the_dynamic_per_token_scale():
    x = torch.randn(2, 5, 16)
    zero = torch.tensor(0.0)
    got = fake_quantize_sampled(x, 4, "row_first_n", 16, zero)
    assert torch.allclose(got, fake_quantize(x, 4, dim=-1), atol=1e-6)


def test_first_row_scale_saturates_later_rows_and_teaches_the_margin():
    x = torch.tensor([[[1.0, -0.4], [3.0, 0.2]]])  # row 0 max 1 -> step 1/7
    log_margin = torch.tensor(0.0, requires_grad=True)
    y = fake_quantize_sampled(x, 4, "first_row", 0, log_margin)
    assert torch.allclose(y[0, 0], torch.tensor([1.0, -3 / 7]))
    assert torch.isclose(y[0, 1, 0], torch.tensor(1.0))  # 3.0 clipped at row 0's max
    y.sum().backward()
    assert log_margin.grad is not None and log_margin.grad != 0


def test_sampled_qat_learns_margins_and_ptq_reproduces_it():
    # Untied: PTQ rounds a tied table for the input lookup too, which QAT never sees.
    cfg = tiny(**{"model.tie_embeddings": False})
    model = Bevo(cfg)
    _, a_bits, kv_bits = SCHEME_BITS["w4a4kv4"]
    margins = start_margins(model, 1.25, a_bits, kv_bits)
    assert margins["blocks.0.attn.kv_quant.k_log_margin"].shape == (1, cfg.model.n_kv_head, 1, 1)
    plain_keys = set(Bevo(cfg).state_dict())
    for sampling in (("first_row", 128), ("first_n", 16), ("row_first_n", 8), ("first_tile", 8)):
        model = Bevo(cfg)
        apply_qat(model, "w4a4kv4", margins, sampling)
        x = torch.randint(
            0, cfg.tokenizer.vocab_size, (2, 32), generator=torch.Generator().manual_seed(0)
        )
        _, loss, _ = model(x, x)
        loss.backward()
        learned = {n: p for n, p in model.named_parameters() if n.endswith("_log_margin")}
        assert len(learned) == len(margins)
        assert any(p.grad.abs().sum() > 0 for p in learned.values())

        groups = model.configure_optimizer(1e-3, 0.1).param_groups
        scale_group = [g for g in groups if g.get("lr_scale")]
        assert {id(p) for p in scale_group[0]["params"]} == {id(p) for p in learned.values()}

        weights, saved = split_scales(model.state_dict())
        assert set(weights) == plain_keys and set(saved) == set(margins)
        model.eval()
        with torch.no_grad():
            qat_logits, _, _ = model(x)
            plain = Bevo(cfg)
            plain.load_state_dict(weights)
            apply_ptq(plain.eval(), "w4a4kv4", saved, sampling)
            ptq_logits, _, _ = plain(x)
        assert torch.allclose(qat_logits, ptq_logits, atol=1e-5)


def test_carried_max_keeps_the_first_rows_sample_across_calls():
    x = torch.randn(2, 3, 10, 6)  # (B, H, T, C)
    for mode, n in (("first_row", 0), ("first_n", 2), ("first_tile", 4)):
        whole = sample_max(x, mode, n)
        memory = ScaleMemory()
        assert torch.equal(carried_max(x[..., :5, :], mode, n, memory, "k"), whole)
        for t in range(5, 10):  # one row per call: the first rows' max, not the row's own
            assert torch.equal(carried_max(x[..., t : t + 1, :], mode, n, memory, "k"), whole)
        assert memory["k"][0] == 10
    memory = ScaleMemory()  # a prompt shorter than the tile: rows 0..3 build the max
    got = [carried_max(x[..., t : t + 1, :], "first_tile", 4, memory, "k") for t in range(6)]
    for t in range(6):
        assert torch.equal(got[t], sample_max(x[..., : min(t, 3) + 1, :], "first_tile", 4))
    per_row = carried_max(x, "row_first_n", 2, ScaleMemory(), "k")
    assert torch.equal(per_row, sample_max(x, "row_first_n", 2))  # needs no memory


@pytest.mark.parametrize("mode,n", [("first_row", 128), ("first_n", 4), ("first_tile", 4)])
def test_cached_generation_matches_the_whole_sequence(mode, n):
    """Token-by-token decoding after a prompt that covers the sample gives the logits of
    one whole-sequence call: every token uses the first rows' scale either way."""
    cfg = tiny()
    torch.manual_seed(0)
    model = Bevo(cfg).double().eval()
    _, a_bits, kv_bits = SCHEME_BITS["w4a4kv4"]
    margins = start_margins(model, 1.25, a_bits, kv_bits)
    apply_ptq(model, "w4a4kv4", {k: v.double() for k, v in margins.items()}, (mode, n))
    idx = torch.randint(
        0, cfg.tokenizer.vocab_size, (2, 20), generator=torch.Generator().manual_seed(1)
    )
    prompt = 6
    with torch.no_grad():
        whole, _, _ = model(idx)
        logits, _, caches = model(idx[:, :prompt])
        steps = [logits[:, -1]]
        for t in range(prompt, 19):
            logits, _, caches = model(idx[:, t : t + 1], caches=caches)
            steps.append(logits[:, -1])
    decoded = torch.stack(steps, dim=1)
    assert torch.allclose(decoded, whole[:, prompt - 1 : 19], atol=1e-9)
