"""Integer-only engine: no floating-point op at inference, exact integer helpers, an exact
KV cache, and logits that track the float model, with static and with sampled scales."""

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

from bevotron.config import RunConfig, apply_overrides, load_raw  # noqa: E402
from bevotron.models import Bevo  # noqa: E402
from bevotron.quant import int_engine  # noqa: E402
from bevotron.quant.hybrid import start_margins  # noqa: E402
from bevotron.quant.int_export import IntSpec, build, collect, sampling_multiplier  # noqa: E402
from bevotron.quant.ptq import apply_ptq  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


class NoFloat(TorchDispatchMode):
    """Fails on any op that reads or writes a floating-point tensor."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten((args, kwargs, out))[0]:
            if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex()):
                raise AssertionError(f"floating-point tensor in {func}")
        return out


def tiny_model(**overrides):
    base = {
        "model.n_layer": 2,
        "model.d_model": 64,
        "model.n_head": 4,
        "model.n_kv_head": 4,
        "model.context_len": 32,
        "tokenizer.vocab_size": 300,
    }
    cfg = RunConfig.from_dict(apply_overrides(load_raw(BASELINE), {**base, **overrides}))
    torch.manual_seed(0)
    return Bevo(cfg).eval()


def batches(n=4, t=32, vocab=300):
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, vocab, (2, t), generator=gen) for _ in range(n)]


def export(model, spec):
    return build(model, spec, collect(model, batches()))


def test_forward_and_generation_use_no_floating_point():
    with pytest.raises(AssertionError), NoFloat():  # the guard itself works
        torch.ones(2, dtype=torch.int64) * 1.5
    P = export(tiny_model(), IntSpec())
    x = batches(1)[0]
    temp = sampling_multiplier(P, 0.7)  # set up outside the engine, like the export
    gen = torch.Generator().manual_seed(0)
    with NoFloat():
        logits, _ = int_engine.forward(P, x)
        greedy = int_engine.generate(P, x[:, :4], 6)
        sampled = int_engine.generate(P, x[:, :4], 6, temp=temp, top_k=20, gen=gen)
    assert logits.dtype == torch.int64 and greedy.shape == sampled.shape == (2, 10)


def test_every_exported_tensor_is_an_integer():
    P = export(tiny_model(), IntSpec())
    leaves = [t for t in tree_flatten({k: v for k, v in P.items() if k != "meta"})[0]]
    assert all(not torch.is_tensor(t) or not t.is_floating_point() for t in leaves)


def test_requant_rounds_to_nearest_for_both_signs():
    x = torch.tensor([5, -5, 6, -6, 7, -7])
    # x * 3 / 4 = 3.75, -3.75, 4.5, -4.5, 5.25, -5.25 (halves round up)
    y = int_engine.requant(x, torch.tensor(3), torch.tensor(2), -100, 100)
    assert y.tolist() == [4, -4, 5, -4, 5, -5]
    assert int_engine.requant(torch.tensor([1000]), 1, 1, -8, 7).tolist() == [7]


def test_isqrt_is_floor_sqrt():
    v = torch.tensor([1, 2, 3, 4, 15, 16, 17, 10**12, 10**12 - 1, (1 << 62) - 1])
    want = [int(__import__("math").isqrt(int(a))) for a in v]
    assert int_engine.isqrt(v).tolist() == want


def test_kv_cache_matches_full_forward_exactly():
    P = export(tiny_model(), IntSpec())
    x = batches(1)[0][:, :12]
    full, _ = int_engine.forward(P, x)
    logits, caches = int_engine.forward(P, x[:, :5])
    steps = [logits]
    for i in range(5, 12):
        logits, caches = int_engine.forward(P, x[:, i : i + 1], caches)
        steps.append(logits)
    assert torch.equal(torch.cat(steps, dim=1), full)


def test_wide_integer_engine_tracks_the_float_model():
    model = tiny_model()
    spec = IntSpec(w_bits=8, a_bits=8, kv_bits=8, out_bits=12, p_bits=12, logit_bits=16)
    P = export(model, spec)
    x = batches(1)[0]
    with torch.no_grad():
        want, _, _ = model(x)
    got, _ = int_engine.forward(P, x)
    got = got.double() * P["meta"]["logit_scale"]
    corr = torch.corrcoef(torch.stack([want.flatten().double(), got.flatten()]))[0, 1]
    assert corr > 0.98
    agree = (want.argmax(-1) == got.argmax(-1)).double().mean()
    assert agree > 0.8


SAMPLED = (("first_row", 128), ("first_n", 16), ("row_first_n", 8), ("first_tile", 8))


def test_quant_sampled_rounds_like_the_float_formula():
    x = torch.tensor([[[100, -40, 7, 0], [250, -300, 13, -1]]])  # (B, T, C) holding values
    mm = torch.tensor(5120)  # margin 1.25 over 2^12
    cases = (  # mode, n, the sampled max it must find
        ("first_row", 0, [[[100]]]),
        ("first_n", 2, [[[100]]]),
        ("row_first_n", 2, [[[100], [300]]]),
        ("first_tile", 2, [[[300]]]),  # rows 0-1, values 0-1: 100, -40, 250, -300
    )
    for mode, n, top in cases:
        codes, a = int_engine.quant_sampled(x, mode, n, mm, 12, -8, 7)
        assert a.tolist() == top
        amax = a.double()
        want = x.double() * 7 / (amax * 1.25)
        want = (want.abs() + 0.5).floor() * want.sign()  # halves away from zero
        assert torch.equal(codes, want.clamp(-8, 7).long())


@pytest.mark.parametrize("mode,n", SAMPLED)
def test_sampled_engine_uses_no_floating_point(mode, n):
    P = export(tiny_model(), IntSpec(scales=mode, scale_n=n, scale_margin=1.25))
    assert P["consts"]["mode"] == mode
    leaves = tree_flatten({k: v for k, v in P.items() if k != "meta"})[0]
    assert all(not torch.is_tensor(t) or not t.is_floating_point() for t in leaves)
    x = batches(1)[0]
    with NoFloat():
        logits, caches = int_engine.forward(P, x)
        greedy = int_engine.generate(P, x[:, :4], 6)
    assert logits.dtype == torch.int64 and greedy.shape == (2, 10)
    k, v, ak, av = caches[0]
    assert ak.shape == av.shape == (*k.shape[:3], 1)  # one runtime factor per cached row


def test_row_sampled_kv_cache_matches_full_forward_exactly():
    # Per-row scales do not depend on other rows, so cached decoding is exact. (first_row
    # and first_n give every new token its own one-row matrix, so they differ by design.)
    P = export(tiny_model(), IntSpec(scales="row_first_n", scale_n=8))
    x = batches(1)[0][:, :12]
    full, _ = int_engine.forward(P, x)
    logits, caches = int_engine.forward(P, x[:, :5])
    steps = [logits]
    for i in range(5, 12):
        logits, caches = int_engine.forward(P, x[:, i : i + 1], caches)
        steps.append(logits)
    assert torch.equal(torch.cat(steps, dim=1), full)


@pytest.mark.parametrize("mode,n", SAMPLED)
def test_wide_sampled_engine_tracks_the_float_model(mode, n):
    # Against the float model with the same sampled scales at 8 bits (fake quant).
    model = tiny_model()
    spec = IntSpec(
        w_bits=8, a_bits=8, kv_bits=8, out_bits=12, p_bits=12, logit_bits=16,
        scales=mode, scale_n=n, scale_margin=1.25,
    )  # fmt: skip
    P = export(model, spec)
    sim = copy.deepcopy(model)
    apply_ptq(sim, "int8", start_margins(sim, 1.25, 8, 8), (mode, n))
    x = batches(1)[0]
    with torch.no_grad():
        want, _, _ = sim(x)
    got, _ = int_engine.forward(P, x)
    got = got.double() * P["meta"]["logit_scale"]
    corr = torch.corrcoef(torch.stack([want.flatten().double(), got.flatten()]))[0, 1]
    assert corr > 0.98
