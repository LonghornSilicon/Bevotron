"""Integer-only inference for bevo.

Every tensor the forward pass touches is an integer. There is no floating-point op at
inference time:

  matmul      int32 accumulate (optionally saturated to acc_bits)
  rescale     integer multiply, then a rounding right shift ("requantization"); every
              real-valued scale was folded into a (multiplier, shift) pair offline
  LayerNorm   integer mean and variance, integer square root, integer division
  softmax     exp lookup table on (row max - score), integer sum, integer division
  GELU        lookup table from the up projection's output code to down's input code
  logits      integers; greedy decoding is an integer argmax, sampling uses the exp table
              and an integer random draw
  sampled     (first_row, first_n, row_first_n, first_tile; quant.hybrid) a Linear input
              or KV head is
  scales      written at mid_bits with a fixed scale, the max of a sample of the matrix
              is a runtime integer a, codes are round(x * qmax / (a * margin)) by integer
              division, and the consumer multiplies its accumulator by a before its
              static requantization. Cached keys and values keep a per row.
              Cached generation carries each whole-matrix a across steps (the
              sequence's first rows set it), the last entry of every layer's cache.

quant.int_export builds the integer model (the only place floats appear, once, offline).
tests/test_int_engine.py runs this forward pass under a mode that fails on any
floating-point op.

Shapes: B batch, T new tokens, Tk cached + new tokens, d model width, H heads, hd head
width. Values are held in int64 tensors; their ranges are set by the IntSpec bit widths:

  residual    res_bits, one fixed scale for the whole model
  q           out_bits;  k, v (the cache) kv_bits, one scale per head
  linear in   a_bits (from LayerNorm, attention, GELU table)
  linear out  out_bits (o, up, down), logit_bits (head)
  p           p_bits unsigned (attention probabilities, scale 2^-p_bits)
"""

from __future__ import annotations

import torch

from bevotron.quant.hybrid import ScaleMemory, carried_max, sample_max

INT32 = torch.int32


def requant(x: torch.Tensor, m, s, lo: int, hi: int) -> torch.Tensor:
    """round(x * m / 2^s) clamped to [lo, hi]. m and s are int64 (broadcastable), s >= 1."""
    s = torch.as_tensor(s)
    half = torch.bitwise_left_shift(torch.ones_like(s), s - 1)
    return torch.bitwise_right_shift(x * m + half, s).clamp(lo, hi)


def _int8_digits(x: torch.Tensor) -> list[torch.Tensor]:
    """x as int8 digits, x = sum(d_i * 128^i): low digits in [0, 127], the top one signed."""
    digits = []
    while x.abs().max() > 127:
        digits.append(torch.bitwise_and(x, 127).to(torch.int8))
        x = torch.bitwise_right_shift(x, 7)
    return digits + [x.to(torch.int8)]


def _mm_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact integer (m,k) x (k,n) on CUDA int8 tensor cores (torch._int_mm, int32 out).
    Operands wider than int8 are split into int8 digits; shapes are zero-padded to what
    _int_mm accepts (m > 16, k and n multiples of 8)."""
    m, k = a.shape
    n = b.shape[1]
    pm, pk, pn = max(32, -(-m // 8) * 8), -(-k // 8) * 8, -(-n // 8) * 8
    a = torch.nn.functional.pad(a, (0, pk - k, 0, pm - m))
    b = torch.nn.functional.pad(b, (0, pn - n, 0, pk - k))
    out = torch.zeros(pm, pn, dtype=torch.int64, device=a.device)
    for i, da in enumerate(_int8_digits(a)):
        for j, db in enumerate(_int8_digits(b)):
            part = torch._int_mm(da.contiguous(), db.contiguous()).to(torch.int64)
            out += torch.bitwise_left_shift(part, 7 * (i + j))
    return out[:m, :n]


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.device.type != "cuda":
        return torch.matmul(a.to(INT32), b.to(INT32)).to(torch.int64)
    if b.dim() == 2:  # (..., k) x (k, n)
        return _mm_cuda(a.reshape(-1, a.shape[-1]), b).reshape(*a.shape[:-1], b.shape[1])
    lead = a.shape[:-2]  # batched: (..., m, k) x (..., k, n), one product per batch entry
    a2, b2 = a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])
    out = torch.stack([_mm_cuda(x, y) for x, y in zip(a2, b2, strict=True)])
    return out.reshape(*lead, *out.shape[-2:])


def imatmul(a: torch.Tensor, b: torch.Tensor, acc_bits: int, stats: dict | None, key: str):
    """Integer matmul with an int32 accumulator, saturated to acc_bits."""
    acc = _matmul(a, b)
    if stats is not None:
        stats[key] = max(stats.get(key, 0), int(acc.abs().max()))
    if acc_bits < 32:
        lim = 1 << (acc_bits - 1)
        acc = acc.clamp(-lim, lim - 1)
    return acc


def linear(x: torch.Tensor, p: dict, acc_bits: int, stats: dict | None, key: str, a=None):
    """x times the integer weights, requantized to the output code. `a` is the runtime
    factor of a sampled input scale (None for static)."""
    acc = imatmul(x, p["w"].T, acc_bits, stats, key)
    if a is not None:
        acc = acc * a
    return requant(acc, p["m"], p["s"], p["lo"], p["hi"])


def quant_sampled(
    x: torch.Tensor,
    mode: str,
    n: int,
    mm,
    mbits: int,
    lo: int,
    hi: int,
    memory: ScaleMemory | None = None,
    key: str = "",
):
    """x (..., T, C) int16 holding values -> (codes, a). a is the max |value| of the
    sample (row 0, row 0's first n values, the top-left n x n tile, or each row's first n),
    kept (..., 1, 1) or (..., T, 1), and carried across the calls of a sequence when a
    memory is given (quant.hybrid.carried_max). codes = round(x * hi / (a * mm / 2^mbits)),
    clamped: one integer division per value, no float."""
    a = sample_max(x, mode, n) if memory is None else carried_max(x, mode, n, memory, key)
    a = a.clamp(min=1)
    den = a * mm
    num = torch.bitwise_left_shift(x.abs() * hi, mbits + 1) + den
    q = torch.div(num, 2 * den, rounding_mode="floor")  # nearest, halves away from zero
    return (q * x.sign()).clamp(lo, hi), a


def act_in(x: torch.Tensor, p: dict, c: dict, memory: ScaleMemory | None = None, key=""):
    """A Linear's input codes and runtime factor: as written (static, a = None), or
    sampled from the int16 holding values."""
    if c["mode"] is None:
        return x, None
    mode, n = c["mode"], c["scale_n"]
    return quant_sampled(x, mode, n, p["mm"], c["margin_bits"], c["a_lo"], c["a_hi"], memory, key)


def isqrt(v: torch.Tensor) -> torch.Tensor:
    """floor(sqrt(v)) for int64 v >= 1, elementwise (Newton's method from above)."""
    y = torch.full_like(v, 1 << 32)
    for _ in range(64):
        y = torch.minimum(
            y, torch.bitwise_right_shift(y + torch.div(v, y, rounding_mode="floor"), 1)
        )
    return y


def layernorm(x: torch.Tensor, p: dict, frac_bits: int, std_bits: int) -> torch.Tensor:
    """(x - mean) / std * gamma + beta, as an integer at scale 2^-frac_bits * gamma_scale.
    Consumers requantize it to their own input scale."""
    d = x.shape[-1]
    mean = torch.div(x.sum(-1, keepdim=True) + d // 2, d, rounding_mode="floor")
    xc = x - mean
    var = torch.div((xc * xc).sum(-1, keepdim=True), d, rounding_mode="floor") + p["eps"]
    std = isqrt(torch.bitwise_left_shift(var, 2 * std_bits))  # sqrt(var) * 2^std_bits
    num = torch.bitwise_left_shift(xc, frac_bits + std_bits) + torch.div(
        std, 2, rounding_mode="floor"
    )
    n = torch.div(num, std, rounding_mode="floor")  # (x - mean) / std * 2^frac_bits
    return n * p["g"] + p["b"]


def causal_mask(t: int, tk: int, window: int | None, device) -> torch.Tensor:
    q_pos = torch.arange(tk - t, tk, device=device)[:, None]
    k_pos = torch.arange(tk, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed &= q_pos - k_pos < window
    return allowed


def attention(q, k, v, ak, av, L: dict, P: dict, stats: dict | None, window: int | None):
    """q (B,H,T,hd) out_bits; k, v (B,H,Tk,hd) kv_bits; ak, av (B,H,Tk,1) the runtime
    factors of sampled K and V scales, or None. Returns (B,T,d) written for o's input
    (its codes, or int16 holding values when sampled)."""
    c = P["consts"]
    t, tk = q.shape[2], k.shape[2]
    mask = causal_mask(t, tk, window, q.device)[None, None]
    scores = imatmul(q, k.transpose(-1, -2), c["acc_bits"], stats, "scores")
    if ak is not None:  # each key column at its own scale: bring them to one
        scores = scores * ak.transpose(-1, -2)
    top = scores.masked_fill(~mask, torch.iinfo(torch.int64).min).amax(-1, keepdim=True)
    idx = requant(top - scores, L["score_m"], L["score_s"], 0, P["exp_lut"].numel() - 1)
    e = P["exp_lut"][idx] * mask
    total = e.sum(-1, keepdim=True)
    p_bits = c["p_bits"]
    p = torch.div(torch.bitwise_left_shift(e, p_bits) + total // 2, total, rounding_mode="floor")
    p = p.clamp(0, (1 << p_bits) - 1)
    if av is not None:  # fold each value row's factor into its probability
        p = p * av.transpose(-1, -2)  # up to p_bits + mid_bits - 1 bits
        p = requant(p, 1, c["pv_shift"], 0, torch.iinfo(torch.int64).max)  # to PV_FOLD_BITS
    pv = imatmul(p, v, c["acc_bits"], stats, "pv")  # (B,H,T,hd)
    y = requant(pv, L["pv_m"], L["pv_s"], c["in_lo"], c["in_hi"])
    b, h, _, hd = y.shape
    return y.transpose(1, 2).reshape(b, t, h * hd)


def forward(P: dict, idx: torch.Tensor, caches=None, stats: dict | None = None):
    """idx (B,T) int64 token ids -> (logits (B,T,V) int64 at the logit scale, caches).
    Each layer's cache is (k, v, ak, av, memory): memory holds the sampled maxima carried
    across calls, one object shared by every layer."""
    c, meta = P["consts"], P["meta"]
    acc = c["acc_bits"]
    r_lo, r_hi = c["res_lo"], c["res_hi"]
    frac, std_bits = c["ln_frac_bits"], c["ln_std_bits"]
    n_head, n_kv, hd = meta["n_head"], meta["n_kv_head"], meta["head_dim"]
    in_lo, in_hi, mode = c["in_lo"], c["in_hi"], c["mode"]
    b, t = idx.shape
    offset = 0 if caches is None else caches[0][0].shape[2]
    memory = ScaleMemory() if caches is None else caches[0][4]

    emb = P["emb"]
    r = requant(emb["q"][idx], emb["m"][idx][..., None], emb["s"][idx][..., None], r_lo, r_hi)
    pos = P["pos"]
    r = (r + requant(pos["q"][offset : offset + t], pos["m"], pos["s"], r_lo, r_hi)).clamp(
        r_lo, r_hi
    )

    new_caches = []
    for i, L in enumerate(P["layers"]):
        h = layernorm(r, L["ln1"], frac, std_bits)
        to = L["ln1"]["to"]
        x = {
            n: act_in(requant(h, *to[n], in_lo, in_hi), L[n], c, memory, f"{i}.{n}")
            for n in ("q", "k", "v")
        }
        q = linear(x["q"][0], L["q"], acc, stats, "q", x["q"][1])
        q = q.view(b, t, n_head, hd).transpose(1, 2)
        kv = []
        for n in ("k", "v"):
            y = linear(x[n][0], L[n], acc, stats, n, x[n][1]).view(b, t, n_kv, hd).transpose(1, 2)
            a = None
            if mode is not None:  # sampled per head, the factor kept for every cached row
                lo, hi = c["kv_lo"], c["kv_hi"]
                mm, key = L[f"{n}_mm"], f"{i}.{n}_cache"
                y, a = quant_sampled(
                    y, mode, c["scale_n"], mm, c["margin_bits"], lo, hi, memory, key
                )
                a = a.expand(b, n_kv, t, 1)
            kv += [y, a]
        k, ak, v, av = kv
        if caches is not None:
            k = torch.cat((caches[i][0], k), dim=2)
            v = torch.cat((caches[i][1], v), dim=2)
            if mode is not None:
                ak = torch.cat((caches[i][2], ak), dim=2)
                av = torch.cat((caches[i][3], av), dim=2)
        new_caches.append((k, v, ak, av, memory))
        if n_kv != n_head:
            rep = n_head // n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            if mode is not None:
                ak, av = ak.repeat_interleave(rep, dim=1), av.repeat_interleave(rep, dim=1)
        window = meta["layer_windows"][i]
        y = attention(q, k, v, ak, av, L, P, stats, window)
        x_o, a_o = act_in(y, L["o"], c, memory, f"{i}.o")
        o = linear(x_o, L["o"], acc, stats, "o", a_o)
        r = (r + requant(o, *L["res_attn"], r_lo, r_hi)).clamp(r_lo, r_hi)

        h = layernorm(r, L["ln2"], frac, std_bits)
        h = requant(h, *L["ln2"]["to"]["up"], in_lo, in_hi)
        x_up, a_up = act_in(h, L["up"], c, memory, f"{i}.up")
        u = linear(x_up, L["up"], acc, stats, "up", a_up)
        x_dn, a_dn = act_in(L["gelu_lut"][u - L["up"]["lo"]], L["down"], c, memory, f"{i}.down")
        dn = linear(x_dn, L["down"], acc, stats, "down", a_dn)
        r = (r + requant(dn, *L["res_mlp"], r_lo, r_hi)).clamp(r_lo, r_hi)
        if stats is not None:
            stats["residual"] = max(stats.get("residual", 0), int(r.abs().max()))

    h = layernorm(r, P["lnf"], frac, std_bits)
    h = requant(h, *P["lnf"]["to"]["head"], in_lo, in_hi)
    x, a = act_in(h, P["head"], c, memory, "head")
    return linear(x, P["head"], acc, stats, "head", a), new_caches


def sample_next(P: dict, logits: torch.Tensor, temp: tuple | None, top_k: int | None, gen):
    """logits (B,V) int64. temp None = greedy (integer argmax). Otherwise temp is the
    (multiplier, shift) that maps a logit gap to an exp-table index at that temperature
    (quant.int_export.sampling_multiplier); the draw is an integer in [0, total)."""
    if temp is None:
        return logits.argmax(-1, keepdim=True)
    top = logits.amax(-1, keepdim=True)
    idx = requant(top - logits, temp[0], temp[1], 0, P["exp_lut"].numel() - 1)
    e = P["exp_lut"][idx]
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        e = e * (logits >= kth)
    cum = e.cumsum(-1)
    draw = torch.stack([torch.randint(0, int(t), (1,), generator=gen) for t in cum[:, -1]])
    draw = draw.to(cum.device)
    return torch.searchsorted(cum, draw, right=True)


def generate(P: dict, idx: torch.Tensor, max_new_tokens: int, temp=None, top_k=None, gen=None):
    """Autoregressive decoding with an integer KV cache. idx (B,T) int64."""
    ctx = P["meta"]["context_len"]
    caches, out = None, idx
    for _ in range(max_new_tokens):
        step = out if caches is None else out[:, -1:]
        if out.shape[1] > ctx:  # cache would run past the position table: restart the window
            caches, step = None, out[:, -ctx:]
        logits, caches = forward(P, step, caches)
        out = torch.cat((out, sample_next(P, logits[:, -1], temp, top_k, gen)), dim=1)
    return out
