"""Integer-only engine: no floating-point op at inference, exact integer helpers, an exact
KV cache, and logits that track the float model."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

from bevotron.config import RunConfig, apply_overrides, load_raw  # noqa: E402
from bevotron.models import Bevo  # noqa: E402
from bevotron.quant import int_engine  # noqa: E402
from bevotron.quant.int_export import IntSpec, build, collect, sampling_multiplier  # noqa: E402

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
