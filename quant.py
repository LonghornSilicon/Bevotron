"""
Simulated ("fake") quantization: round weights to a low-precision number format and
straight back to float. Nothing gets faster or smaller; the point is to measure what a
format costs in accuracy, for any format (including the ones our hardware will use),
on any device.

Formats are strings:
    'int8', 'int4'   symmetric integers, one scale per output channel (row of the weight)
    'int4_g64'       same, but one scale per group of 64 consecutive input channels
A new format (fp8, fixed point, ...) is a new branch in fake_quant.

Two ways to use it:
1) quantization-aware training: GPTConfig.quant = 'int4_g64' builds the transformer
   blocks with QuantLinear, which trains against quantized weights
2) post-training quantization: quantize_(model.transformer.h, 'int4_g64') rounds the
   weights of a trained model in place
Embeddings and the (tied) lm_head stay in full precision in both cases.
"""

import re

import torch
import torch.nn as nn
from torch.nn import functional as F

def fake_quant(w, fmt):
    """Round a (out_features, in_features) weight to fmt and back, keeping its shape and dtype."""
    m = re.fullmatch(r'int(\d+)(?:_g(\d+))?', fmt)
    assert m, f"unknown quantization format: {fmt}"
    bits, group = int(m[1]), int(m[2]) if m[2] else w.size(-1)
    assert bits >= 2 and w.size(-1) % group == 0
    qmax = 2 ** (bits - 1) - 1 # e.g. int4 -> integers in [-7, 7]
    g = w.reshape(-1, group) # each row is one group of weights sharing a scale
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    return ((g / scale).round().clamp(-qmax, qmax) * scale).reshape(w.shape)

class QuantLinear(nn.Linear):
    """nn.Linear that runs its forward pass with fake-quantized weights."""

    def __init__(self, in_features, out_features, bias=True, fmt='int8'):
        super().__init__(in_features, out_features, bias=bias)
        self.fmt = fmt

    def forward(self, x):
        # straight-through estimator: quantized weights forward, gradients as if unquantized
        w = self.weight + (fake_quant(self.weight, self.fmt) - self.weight).detach()
        return F.linear(x, w, self.bias)

@torch.no_grad()
def quantize_(module, fmt):
    """Post-training quantization: round the weight of every nn.Linear inside module, in place."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.copy_(fake_quant(m.weight, fmt))
    return module
