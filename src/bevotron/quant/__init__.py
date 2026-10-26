"""Quantization (simulated): PTQ measures the accuracy cost of a scheme; QAT trains against it."""

from bevotron.quant.ptq import apply_ptq, fake_quantize
from bevotron.quant.qat import apply_qat

__all__ = ["apply_ptq", "apply_qat", "fake_quantize"]
