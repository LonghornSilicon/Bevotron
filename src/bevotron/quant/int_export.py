"""Build the integer model that quant.int_engine runs, and score it.

This is the offline step, and the only place floats appear: it measures fixed scales on
calibration text (or takes the scales a static-QAT run learned), rounds the weights, and
folds every ratio of scales into an integer (multiplier, shift) pair, so that the engine
needs nothing but integer multiplies, shifts, divisions and table lookups.

With sampled scales (IntSpec.scales first_row, first_n, row_first_n or first_tile, see
quant.hybrid)
each Linear input and each KV head is first written at mid_bits (int16) with a fixed
scale, the max of a sample of that matrix becomes a runtime integer `a`, and the codes are
round(x * qmax / (a * margin)) by integer division. The real scale of a code is then a
times a fixed unit, so every consumer's (multiplier, shift) is static and the engine
multiplies the accumulator by `a` before requantizing.

The scoring helpers (`int_eval`) turn the engine's integer logits into a cross-entropy
loss in float. That is measurement, outside the engine, in the same way a test bench
reads a chip's outputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from bevotron.quant.ptq import SCHEME_BITS
from bevotron.quant.static import CALIB_PERCENTILE, SAMPLES_PER_BATCH

RES_MARGIN = 1.25  # residual and logit scales cover 1.25x the largest value seen
LN_FRAC_BITS = 10  # LayerNorm's normalized value carries 10 fractional bits
LN_STD_BITS = 8  # extra bits on the integer square root
LN_GAMMA_BITS = 16
EXP_BITS = 16  # exp table entries are round(exp(-i / lut_res) * 2^16)
MARGIN_BITS = 12  # sampled-scale headroom as an integer over 2^12
SAMPLED_M_BITS = 15  # multipliers that also take a runtime factor: acc * a * m fits int64
# Sampled V rows: probability times the row's factor, shifted down to this many bits, so
# the attention-value sum stays within an int16 accumulator.
PV_FOLD_BITS = 12


@dataclass(frozen=True)
class IntSpec:
    """Bit widths of the integer engine. Sweep these."""

    w_bits: int = 4  # weights, one scale per output channel
    a_bits: int = 4  # every Linear input
    kv_bits: int = 4  # KV cache, one scale per head
    out_bits: int = 8  # q, o, up and down outputs
    acc_bits: int = 32  # matmul accumulator (saturating below 32)
    p_bits: int = 8  # attention probabilities, unsigned
    res_bits: int = 16  # residual stream
    logit_bits: int = 16
    pos_bits: int = 8  # learned position table
    lut_res: int = 32  # exp table steps per unit of score gap
    lut_size: int = 512  # exp table length; gaps past lut_size / lut_res count as 0
    # static: fixed scales. first_row | first_n | row_first_n | first_tile: sampled
    scales: str = "static"
    scale_n: int = 128  # values sampled by first_n and row_first_n, tile side for first_tile
    scale_margin: float = 1.0  # headroom when the run has no learned margins
    mid_bits: int = 16  # holding width before a sampled quantization

    @classmethod
    def for_scheme(cls, scheme: str, **overrides) -> IntSpec:
        w, a, kv = SCHEME_BITS[scheme]
        if None in (w, a, kv):
            raise ValueError(f"scheme {scheme!r} leaves part of the model in float")
        return cls(**{"w_bits": w, "a_bits": a, "kv_bits": kv, **overrides})


def _range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def multiplier(ratio, bits: int = 31) -> tuple[torch.Tensor, torch.Tensor]:
    """A positive real ratio as int64 (m, s) with ratio ~= m / 2^s, m in [2^(bits-1), 2^bits)."""
    r = torch.as_tensor(ratio, dtype=torch.float64).clamp(min=1e-30)
    s = (bits - 1 - torch.floor(torch.log2(r))).clamp(1, 62).to(torch.int64)
    m = torch.round(r * torch.pow(2.0, s.to(torch.float64))).to(torch.int64)
    return m, s


def _quantize_rows(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric, one scale per row. Returns (int codes, float64 scales)."""
    w = w.detach().double()
    scale = (w.abs().amax(dim=1) / _qmax(bits)).clamp(min=1e-12)
    lo, hi = _range(bits)
    return torch.clamp(torch.round(w / scale[:, None]), lo, hi).to(torch.int64), scale


@torch.no_grad()
def collect(model: nn.Module, batches) -> dict:
    """One float pass over calibration batches: percentiles of every Linear's input and
    output, per-head K and V, and the largest residual and logit values."""
    gen = torch.Generator().manual_seed(0)
    samples: dict[str, list] = defaultdict(list)
    peaks: dict[str, float] = defaultdict(float)

    def keep(key, t):
        flat = t.detach().abs().flatten().double()
        peaks[key] = max(peaks[key], float(flat.max()))
        if flat.numel() > SAMPLES_PER_BATCH:
            flat = flat[torch.randint(0, flat.numel(), (SAMPLES_PER_BATCH,), generator=gen)]
        samples[key].append(flat)

    handles, attentions = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(
                mod.register_forward_pre_hook(lambda m, i, n=name: keep(f"{n}.in", i[0]))
            )
            handles.append(mod.register_forward_hook(lambda m, i, o, n=name: keep(f"{n}.out", o)))
        if isinstance(mod, nn.LayerNorm):
            handles.append(mod.register_forward_pre_hook(lambda m, i: keep("residual", i[0])))
        if hasattr(mod, "kv_quant"):

            def record(k, v, n=name):
                for which, t in (("k", k), ("v", v)):
                    for h in range(t.shape[1]):
                        keep(f"{n}.{which}.{h}", t[:, h])
                return k, v

            mod.kv_quant = record
            attentions.append(mod)
    try:
        for x in batches:
            model(x)
    finally:
        for h in handles:
            h.remove()
        for mod in attentions:
            mod.kv_quant = None

    stats = {"peak": dict(peaks)}
    stats["pct"] = {
        k: float(torch.quantile(torch.cat(v).float(), CALIB_PERCENTILE)) for k, v in samples.items()
    }
    return stats


@torch.no_grad()
def build(model: nn.Module, spec: IntSpec, stats: dict, learned: dict | None = None) -> dict:
    """The integer model for `model` (a float bevo with weights loaded). `learned` maps
    quant.static scale names to log scales from a static-QAT checkpoint; they replace the
    calibrated Linear-input and KV scales. With sampled scales (spec.scales) it maps
    margin names to the log margins a sampled-QAT run learned."""
    cfg = model.config.model
    if cfg.norm != "layernorm" or cfg.pos_encoding != "learned" or cfg.mlp_act != "gelu":
        raise NotImplementedError("the integer engine covers the bevo preset only")
    pct, peak = stats["pct"], stats["peak"]
    learned = learned or {}
    a_lo, a_hi = _range(spec.a_bits)
    out_lo, out_hi = _range(spec.out_bits)
    kv_lo, kv_hi = _range(spec.kv_bits)
    res_lo, res_hi = _range(spec.res_bits)
    lg_lo, lg_hi = _range(spec.logit_bits)
    n_head, n_kv, hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    sampled = spec.scales != "static"
    mid_lo, mid_hi = _range(spec.mid_bits)
    in_lo, in_hi = (mid_lo, mid_hi) if sampled else (a_lo, a_hi)  # what feeds a Linear
    m_bits = SAMPLED_M_BITS if sampled else 31
    pv_shift = max(1, spec.p_bits + spec.mid_bits - 1 - PV_FOLD_BITS)

    def margin(key: str) -> torch.Tensor:
        """Learned (or configured) headroom as an integer over 2^MARGIN_BITS."""
        g = learned[key].double().exp() if key in learned else torch.tensor(spec.scale_margin)
        return torch.round(g.double() * 2**MARGIN_BITS).clamp(min=1).to(torch.int64)

    def mid_scale(key) -> float:  # fixed scale of the int16 holding values
        return peak[key] * RES_MARGIN / _qmax(spec.mid_bits)

    def act(name: str) -> tuple[float, float, dict]:
        """(scale of the input codes, per unit of the runtime factor when sampled; scale
        the producer writes at; extra Linear fields)."""
        if not sampled:
            s = in_scale(name)
            return s, s, {}
        s_mid, mm = mid_scale(f"{name}.in"), margin(f"{name}.act_log_margin")
        unit = s_mid * float(mm) / 2**MARGIN_BITS / _qmax(spec.a_bits)
        return unit, s_mid, {"mm": mm}

    def in_scale(name: str) -> float:
        key = f"{name}.act_log_scale"
        if key in learned:
            return math.exp(float(learned[key]))
        return pct[f"{name}.in"] / _qmax(spec.a_bits)

    def kv_scales(attn: str, which: str) -> torch.Tensor:
        key = f"{attn}.kv_quant.{which}_log_scale"
        if key in learned:
            return learned[key].double().exp().flatten()
        p = [pct[f"{attn}.{which}.{h}"] for h in range(n_kv)]
        return torch.tensor(p, dtype=torch.float64) / _qmax(spec.kv_bits)

    s_res = peak["residual"] * RES_MARGIN / _qmax(spec.res_bits)

    def lin(weight, s_in: float, s_out, lo: int, hi: int, extra: dict) -> dict:
        w, s_w = _quantize_rows(weight, spec.w_bits)
        m, s = multiplier(s_in * s_w / torch.as_tensor(s_out, dtype=torch.float64), m_bits)
        return {"w": w, "m": m, "s": s, "lo": lo, "hi": hi, **extra}

    def ln(norm: nn.LayerNorm, consumers: dict[str, float]) -> dict:
        g = norm.weight.detach().double()
        s_g = float(g.abs().max()) / _qmax(LN_GAMMA_BITS)
        unit = 2.0**-LN_FRAC_BITS * s_g  # scale of layernorm()'s output
        return {
            "g": torch.round(g / s_g).to(torch.int64),
            "b": torch.round(norm.bias.detach().double() / unit).to(torch.int64),
            "eps": torch.tensor(max(1, round(norm.eps / s_res**2)), dtype=torch.int64),
            "to": {k: multiplier(unit / v) for k, v in consumers.items()},
        }

    layers = []
    for i, block in enumerate(model.blocks):
        pre, attn = f"blocks.{i}", block.attn
        names = {n: f"{pre}.attn.{n}_proj" for n in ("q", "k", "v", "o")}
        up, down = f"{pre}.mlp.up", f"{pre}.mlp.down"
        s_in, dest, extra = {}, {}, {}  # dest: the scale each Linear's input is written at
        for n, full in [*names.items(), ("up", up), ("down", down)]:
            s_in[n], dest[n], extra[n] = act(full)
        s_q = pct[f"{names['q']}.out"] / _qmax(spec.out_bits)
        per_channel = lambda s: s.repeat_interleave(hd)  # noqa: E731  one scale per head
        kv = {}
        for which in ("k", "v"):
            if sampled:  # written at a fixed int16 scale per head, then sampled
                peaks = [peak[f"{pre}.attn.{which}.{h}"] for h in range(n_kv)]
                s_mid = torch.tensor(peaks, dtype=torch.float64) * RES_MARGIN
                s_mid = s_mid / _qmax(spec.mid_bits)
                mm = margin(f"{pre}.attn.kv_quant.{which}_log_margin").flatten()
                mm = mm.expand(n_kv).clone()
                unit = s_mid * mm.double() / 2**MARGIN_BITS / _qmax(spec.kv_bits)
                kv[which] = (unit, s_mid, mid_lo, mid_hi, {f"{which}_mm": mm.view(1, -1, 1, 1)})
            else:
                s = kv_scales(f"{pre}.attn", which)
                kv[which] = (s, s, kv_lo, kv_hi, {})
        s_k, s_v = kv["k"][0], kv["v"][0]  # per unit of the runtime factor when sampled
        s_o = pct[f"{names['o']}.out"] / _qmax(spec.out_bits)
        s_in_up, s_in_down = s_in["up"], s_in["down"]
        s_up = pct[f"{up}.out"] / _qmax(spec.out_bits)
        s_down = pct[f"{down}.out"] / _qmax(spec.out_bits)
        rep = n_head // n_kv
        s_k_heads, s_v_heads = s_k.repeat_interleave(rep), s_v.repeat_interleave(rep)

        score = s_q * s_k_heads / math.sqrt(hd) * spec.lut_res  # score gap -> exp index
        pv = 2.0**-spec.p_bits * s_v_heads / dest["o"]
        if sampled:  # the engine shifts p * a_v down by pv_shift bits before the matmul
            pv = pv * 2.0**pv_shift
        gelu_in = torch.arange(out_lo, out_hi + 1, dtype=torch.float64) * s_up
        gelu = torch.nn.functional.gelu(gelu_in, approximate="tanh")
        layers.append(
            {
                "ln1": ln(block.norm1, {n: dest[n] for n in ("q", "k", "v")}),
                "q": lin(attn.q_proj.weight, s_in["q"], s_q, out_lo, out_hi, extra["q"]),
                "k": lin(
                    attn.k_proj.weight,
                    s_in["k"],
                    per_channel(kv["k"][1]),
                    *kv["k"][2:4],
                    extra["k"],
                ),
                "v": lin(
                    attn.v_proj.weight,
                    s_in["v"],
                    per_channel(kv["v"][1]),
                    *kv["v"][2:4],
                    extra["v"],
                ),
                "score_m": multiplier(score, m_bits)[0].view(1, -1, 1, 1),
                "score_s": multiplier(score, m_bits)[1].view(1, -1, 1, 1),
                "pv_m": multiplier(pv, m_bits)[0].view(1, -1, 1, 1),
                "pv_s": multiplier(pv, m_bits)[1].view(1, -1, 1, 1),
                "o": lin(attn.o_proj.weight, s_in["o"], s_o, out_lo, out_hi, extra["o"]),
                "res_attn": multiplier(s_o / s_res),
                "ln2": ln(block.norm2, {"up": dest["up"]}),
                "up": lin(block.mlp.up.weight, s_in_up, s_up, out_lo, out_hi, extra["up"]),
                "gelu_lut": torch.clamp(torch.round(gelu / dest["down"]), in_lo, in_hi).to(
                    torch.int64
                ),
                "down": lin(
                    block.mlp.down.weight, s_in_down, s_down, out_lo, out_hi, extra["down"]
                ),
                "res_mlp": multiplier(s_down / s_res),
                **kv["k"][4],
                **kv["v"][4],
            }
        )

    s_head, dest_head, extra_head = act("lm_head")
    s_logit = peak["lm_head.out"] * RES_MARGIN / _qmax(spec.logit_bits)
    head = lin(model.lm_head.weight, s_head, s_logit, lg_lo, lg_hi, extra_head)
    e_q, e_s = _quantize_rows(model.wte.weight, spec.w_bits)
    pos_w = model.positions.table.weight.detach().double()
    s_pos = float(pos_w.abs().max()) / _qmax(spec.pos_bits)
    p_lo, p_hi = _range(spec.pos_bits)
    lut = torch.arange(spec.lut_size, dtype=torch.float64) / spec.lut_res
    return {
        "meta": {
            "n_head": n_head,
            "n_kv_head": n_kv,
            "head_dim": hd,
            "context_len": cfg.context_len,
            "attention_window": cfg.attention_window,
            "spec": asdict(spec),
            # floats below are read only by the scoring and sampling helpers, never the engine
            "logit_scale": s_logit,
            "residual_scale": s_res,
            "learned_scales": bool(learned),
            "scales": spec.scales,
        },
        "consts": {
            "acc_bits": spec.acc_bits,
            "p_bits": spec.p_bits,
            "a_lo": a_lo,
            "a_hi": a_hi,
            "kv_lo": kv_lo,
            "kv_hi": kv_hi,
            "in_lo": in_lo,
            "in_hi": in_hi,
            "mode": spec.scales if sampled else None,
            "scale_n": spec.scale_n,
            "margin_bits": MARGIN_BITS,
            "pv_shift": pv_shift,
            "res_lo": res_lo,
            "res_hi": res_hi,
            "ln_frac_bits": LN_FRAC_BITS,
            "ln_std_bits": LN_STD_BITS,
        },
        "emb": {"q": e_q, **dict(zip(("m", "s"), multiplier(e_s / s_res), strict=True))},
        "pos": {
            "q": torch.clamp(torch.round(pos_w / s_pos), p_lo, p_hi).to(torch.int64),
            **dict(zip(("m", "s"), multiplier(s_pos / s_res), strict=True)),
        },
        "layers": layers,
        "lnf": ln(model.norm_f, {"head": dest_head}),
        "head": head,
        "exp_lut": torch.round(torch.exp(-lut) * 2**EXP_BITS).to(torch.int64),
    }


def to_device(tree, device):
    """The integer model with every tensor moved to the given device."""
    if torch.is_tensor(tree):
        return tree.to(device)
    if isinstance(tree, dict):
        return {k: to_device(v, device) for k, v in tree.items()}
    if isinstance(tree, list | tuple):
        return type(tree)(to_device(v, device) for v in tree)
    return tree


def sampling_multiplier(P: dict, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(m, s) mapping a logit gap to an exp-table index at `temperature`."""
    lut_res = P["meta"]["spec"]["lut_res"]
    return multiplier(P["meta"]["logit_scale"] / max(temperature, 1e-6) * lut_res)


def load_run(run_dir, device: str = "cpu"):
    """(config, float model with the run's weights, learned static scales or {})."""
    from bevotron.config import RunConfig
    from bevotron.models import Bevo
    from bevotron.quant.static import split_scales

    config = RunConfig.load(run_dir / "config.yaml")
    weights, learned = split_scales(torch.load(run_dir / "ckpt.pt", map_location="cpu")["model"])
    model = Bevo(config)
    model.load_state_dict(weights)
    return config, model.to(device).eval(), learned


def export_run(run_dir, spec: IntSpec | None = None, calibrate: bool = False, device="cpu"):
    """The integer model for a trained run. Uses the run's learned static scales (or
    sampled-scale margins) when the run has them, the bit widths match its scheme and
    spec.scales matches its quant.scales, unless calibrate=True."""
    from bevotron.quant.static import calibration_batches

    config, model, learned = load_run(run_dir, device)
    spec = spec or IntSpec.for_scheme(config.quant.scheme)
    run_bits = SCHEME_BITS[config.quant.scheme]
    use = learned and not calibrate and (spec.w_bits, spec.a_bits, spec.kv_bits) == run_bits
    use = use and spec.scales == config.quant.scales
    stats = collect(model, calibration_batches(config, device))
    P = build(model.cpu(), spec, stats, learned if use else None)
    return config, P


@torch.no_grad()
def int_eval(P: dict, data, context_len: int, windows: int, batch: int = 4) -> dict:
    """Mean cross-entropy of the integer engine over the first `windows` validation windows,
    plus the largest accumulator and residual values it produced."""
    from bevotron.quant import int_engine

    stats: dict = {}
    total, count = 0.0, 0
    for start in range(0, windows, batch):
        ids = range(start, min(start + batch, windows))
        chunk = [data[i * context_len : (i + 1) * context_len + 1].astype(np.int64) for i in ids]
        device = P["exp_lut"].device
        xy = torch.from_numpy(np.stack(chunk)).to(device)
        logits, _ = int_engine.forward(P, xy[:, :-1], stats=stats)
        real = logits.double() * P["meta"]["logit_scale"]  # measurement only
        loss = torch.nn.functional.cross_entropy(real.flatten(0, 1), xy[:, 1:].flatten())
        total += float(loss) * xy[:, 1:].numel()
        count += xy[:, 1:].numel()
    return {"val_loss": total / count, "tokens": count, "max_abs": stats}
