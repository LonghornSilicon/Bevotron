"""
Full definition of a GPT-Neo Language Model, all of it in this single file,
written in the style of nanoGPT's model.py. This is the architecture of the
TinyStories checkpoints (roneneldan/TinyStories-*). Differences from GPT-2:
1) layers alternate global and local (sliding-window) causal self-attention
2) attention scores are not scaled by 1/sqrt(head_dim)
3) the q/k/v projections have no bias
References:
1) the official GPT-Neo release by EleutherAI:
https://github.com/EleutherAI/gpt-neo
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py
"""

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config, local):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, value projections for all heads, no bias
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        # local layers only see the previous window_size tokens (themselves included)
        self.window_size = config.window_size if local else None

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # scale=1.0 because GPT-Neo does not scale the scores by 1/sqrt(hs)
        dropout_p = self.dropout if self.training else 0
        if self.window_size is None or T <= self.window_size:
            # global layer, or a window that covers the whole sequence: plain causal, flash attention
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True, scale=1.0)
        else:
            # local layer: query i attends to keys j with i - window_size < j <= i
            i = torch.arange(T, device=x.device)
            dist = i[:, None] - i[None, :]
            mask = (dist >= 0) & (dist < self.window_size)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=1.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # "gelu_new" in Hugging Face
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, local):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, local)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    # defaults are TinyStories-28M
    vocab_size: int = 50257 # GPT-Neo tokenizer
    max_positions: int = 2048 # rows in the position table, i.e. the longest context
    n_layer: int = 8
    n_head: int = 16
    n_embd: int = 512
    window_size: int = 256 # attention span of the local layers
    dropout: float = 0.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_positions, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # even layers are global, odd layers are local
            h = nn.ModuleList([Block(config, local=(i % 2 == 1)) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, same as Hugging Face's GPT-Neo
        self.apply(self._init_weights)

        # report number of parameters (tied weights counted once)
        print("number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters())/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_positions, f"Cannot forward sequence of length {t}, max_positions is only {self.config.max_positions}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT-Neo model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, dropout=0.0):
        """Load GPT-Neo weights from the Hugging Face hub, e.g. 'roneneldan/TinyStories-28M'."""
        from transformers import GPTNeoForCausalLM
        print("loading weights from pretrained gpt-neo: %s" % model_type)
        model_hf = GPTNeoForCausalLM.from_pretrained(model_type)
        c = model_hf.config
        # this file only implements the alternating global/local variant with a 4x gelu MLP
        assert c.attention_layers == ['local' if i % 2 else 'global' for i in range(c.num_layers)]
        assert c.activation_function == 'gelu_new' and c.intermediate_size in (None, 4 * c.hidden_size)
        assert c.layer_norm_epsilon == 1e-5
        config = GPTConfig(vocab_size=c.vocab_size, max_positions=c.max_position_embeddings,
                           n_layer=c.num_layers, n_head=c.num_heads, n_embd=c.hidden_size,
                           window_size=c.window_size, dropout=dropout)
        model = GPT(config)
        # parameter names match Hugging Face's, except attention is one module shallower
        sd_hf = {k.replace('.attn.attention.', '.attn.'): v for k, v in model_hf.state_dict().items()}
        model.load_state_dict(sd_hf)
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_positions
            idx_cond = idx if idx.size(1) <= self.config.max_positions else idx[:, -self.config.max_positions:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
