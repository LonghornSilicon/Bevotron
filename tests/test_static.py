"""Static quantization scales: config, calibration, learned scales in QAT, and the PTQ
path that evaluates them. Need torch; skipped where it is not installed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bevotron.config import RunConfig, apply_overrides, load_raw  # noqa: E402
from bevotron.models import Bevo  # noqa: E402
from bevotron.quant.ptq import SCHEME_BITS, apply_ptq  # noqa: E402
from bevotron.quant.qat import apply_qat  # noqa: E402
from bevotron.quant.static import calibrate, fake_quantize_static, split_scales  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.yaml"


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


def batches(cfg, n=3):
    gen = torch.Generator().manual_seed(0)
    return [torch.randint(0, cfg.tokenizer.vocab_size, (2, 32), generator=gen) for _ in range(n)]


def calibrated(cfg, scheme):
    model = Bevo(cfg)
    _, a_bits, kv_bits = SCHEME_BITS[scheme]
    return model, calibrate(model, batches(cfg), a_bits, kv_bits)


def test_dynamic_default_keeps_earlier_run_ids():
    # Run IDs hash to_dict(); a new default key would rename every run already trained.
    raw = load_raw(BASELINE)
    cfg = RunConfig.from_dict(raw)
    assert "scales" not in cfg.to_dict()["quant"]
    assert cfg.run_id() == RunConfig.from_dict(cfg.to_dict()).run_id()
    static = RunConfig.from_dict(apply_overrides(raw, {"quant.scales": "static"}))
    assert static.to_dict()["quant"]["scales"] == "static"
    assert static.run_id() != cfg.run_id()
    assert RunConfig.from_dict(static.to_dict()).run_id() == static.run_id()
    with pytest.raises(ValueError, match="quant.scales"):
        RunConfig.from_dict(apply_overrides(raw, {"quant.scales": "per-token"}))


def test_static_ptq_shares_the_baseline_checkpoint_but_static_qat_does_not():
    raw = load_raw(BASELINE)
    base = RunConfig.from_dict(raw)
    ptq = RunConfig.from_dict(apply_overrides(raw, {"quant.scales": "static"}))
    qat = {"quant.scheme": "w4a4kv4", "quant.mode": "qat"}
    dyn = RunConfig.from_dict(apply_overrides(raw, qat))
    sta = RunConfig.from_dict(apply_overrides(raw, {**qat, "quant.scales": "static"}))
    assert ptq.training_key() == base.training_key()
    assert sta.training_key() != dyn.training_key()
    assert sta.full_precision_twin().training_key() == base.training_key()


def test_calibration_gives_one_scale_per_linear_and_per_kv_head():
    cfg = tiny()
    model, scales = calibrated(cfg, "w4a4kv4")
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    assert {f"{n}.act_log_scale" for n in linears} <= set(scales)
    k = scales["blocks.0.attn.kv_quant.k_log_scale"]
    assert k.shape == (1, cfg.model.n_kv_head, 1, 1)
    assert all(torch.isfinite(v).all() for v in scales.values())
    assert model.blocks[0].attn.kv_quant is None  # calibration leaves the model plain


def test_static_fake_quant_clips_and_teaches_the_scale():
    log_scale = torch.tensor(0.0, requires_grad=True)  # step 1.0 on a 4-bit grid: [-8, 7]
    x = torch.tensor([0.4, 2.6, 100.0, -100.0], requires_grad=True)
    y = fake_quantize_static(x, 4, log_scale)
    assert torch.equal(y.detach(), torch.tensor([0.0, 3.0, 7.0, -8.0]))
    y.sum().backward()
    assert log_scale.grad is not None and log_scale.grad != 0  # clipped values push the scale
    assert torch.equal(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))  # no gradient past the clip


def test_static_qat_learns_scales_and_ptq_reproduces_it():
    # Untied: PTQ rounds a tied table for the input lookup too, which QAT never sees.
    cfg = tiny(**{"model.tie_embeddings": False})
    model, scales = calibrated(cfg, "w4a4kv4")
    plain_keys = set(Bevo(cfg).state_dict())
    apply_qat(model, "w4a4kv4", scales)
    x = batches(cfg, 1)[0]
    _, loss, _ = model(x, x)
    loss.backward()
    learned = {n: p for n, p in model.named_parameters() if n.endswith("_log_scale")}
    assert learned and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in learned.values()
    )
    assert any(p.grad.abs().sum() > 0 for p in learned.values())

    groups = model.configure_optimizer(1e-3, 0.1).param_groups
    scale_group = [g for g in groups if g.get("lr_scale")]
    assert len(scale_group) == 1 and scale_group[0]["weight_decay"] == 0.0
    assert {id(p) for p in scale_group[0]["params"]} == {id(p) for p in learned.values()}

    weights, saved = split_scales(model.state_dict())
    assert set(weights) == plain_keys  # the weights still load into a plain model
    model.eval()
    with torch.no_grad():
        qat_logits, _, _ = model(x)
        plain = Bevo(cfg)
        plain.load_state_dict(weights)
        apply_ptq(plain.eval(), "w4a4kv4", saved)
        ptq_logits, _, _ = plain(x)
    assert torch.allclose(qat_logits, ptq_logits, atol=1e-5)
