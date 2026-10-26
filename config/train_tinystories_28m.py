# TinyStories-28M architecture (GPT-Neo), trained from scratch on TinyStories.
# Shape matches roneneldan/TinyStories-28M exactly: 8 layers, 512 wide,
# 16 heads, alternating global/local attention (window 256), 2048-row
# position table, GPT-Neo tokenizer (vocab 50257), tied embeddings. ~52.0M params.

model_type = 'gptneo'
out_dir = 'out-tinystories-28m'
dataset = 'tinystories'

# model
n_layer = 8
n_head = 16
n_embd = 512
dropout = 0.0
block_size = 512  # stories are short; position table still has 2048 rows

# batch: 32 x 512 = 16,384 tokens per step
batch_size = 16
gradient_accumulation_steps = 2  # 16 x 2 x 512 = 16,384 tokens per step, fits 16 GB

# optimizer (our choice; the original training recipe is not published in full)
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 500
max_iters = 20000  # ~330M tokens, ~0.7 epoch
lr_decay_iters = 20000
min_lr = 6e-5

# eval / logging
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

compile = False