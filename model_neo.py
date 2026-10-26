"""GPT-Neo with nanoGPT's interface, so train.py runs it unchanged.

Wraps Hugging Face's GPTNeoForCausalLM (the exact architecture of the
TinyStories checkpoints: alternating global/local attention with a 256-token
window, no 1/sqrt(d) attention scaling, tied embeddings) and exposes what
train.py expects: GPTConfig, GPT(config), forward(idx, targets) -> (logits,
loss), configure_optimizers, estimate_mfu, get_num_params, config.block_size.
"""

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPTNeoConfig, GPTNeoForCausalLM


@dataclass
class GPTConfig:
    block_size: int = 512  # training context length (<= max_positions)
    vocab_size: int = 50257  # GPT-Neo tokenizer
    n_layer: int = 8
    n_head: int = 16
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True  # accepted for compatibility; GPT-Neo's biases are fixed
    window_size: int = 256  # local-attention window
    max_positions: int = 2048  # position-table rows, as in TinyStories-28M


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_layer % 2 == 0, "GPT-Neo alternates global/local layers"
        self.config = config
        neo_config = GPTNeoConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_positions,
            hidden_size=config.n_embd,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            attention_types=[[["global", "local"], config.n_layer // 2]],
            window_size=config.window_size,
            intermediate_size=None,  # 4 * hidden_size
            activation_function="gelu_new",
            embed_dropout=config.dropout,
            attention_dropout=config.dropout,
            resid_dropout=config.dropout,
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=True,
        )
        self.neo = GPTNeoForCausalLM(neo_config)
        print("number of parameters: %.2fM" % (self.get_num_params(False) / 1e6,))
        print("  (non-embedding: %.2fM)" % (self.get_num_params() / 1e6,))

    @classmethod
    def from_hf(cls, name, block_size=512):
        """Load a published checkpoint, e.g. 'roneneldan/TinyStories-28M'."""
        hf = GPTNeoForCausalLM.from_pretrained(name)
        c = hf.config
        config = GPTConfig(block_size=block_size, vocab_size=c.vocab_size,
                           n_layer=c.num_layers, n_head=c.num_heads,
                           n_embd=c.hidden_size, window_size=c.window_size,
                           max_positions=c.max_position_embeddings)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config, model.neo = config, hf
        return model

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.neo.transformer.wte.weight.numel()
            n -= self.neo.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        logits = self.neo(input_ids=idx, use_cache=False).logits
        if targets is None:
            return logits[:, [-1], :], None
        # targets are already shifted by get_batch, so no shift here
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.view(-1), ignore_index=-1)
        return logits, loss

    def crop_block_size(self, block_size):
        self.config.block_size = block_size  # position table stays 2048 rows

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = ("fused" in inspect.signature(torch.optim.AdamW).parameters
                 and device_type == "cuda")
        print(f"using fused AdamW: {fused}")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 **({"fused": True} if fused else {}))

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        cfg = self.config
        n = self.get_num_params()
        q = cfg.n_embd // cfg.n_head
        flops_per_token = 6 * n + 12 * cfg.n_layer * cfg.n_head * q * cfg.block_size
        flops = flops_per_token * cfg.block_size * fwdbwd_per_iter / dt
        return flops / 312e12  # nanoGPT's A100 reference; relative only