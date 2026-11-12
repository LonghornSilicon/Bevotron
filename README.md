# Bevotron

Longhorn Silicon's small-scale LLM training pipeline for pre-training and post-training.

One shared baseline config, one small file per study, one command to turn a study into runs.

## Setup 

    uv sync --extra train
    uv run pytest                                            # 64 tests to make sure everything works
    uv run bevotron data export                              # TinyStories train split -> data/tinystories/train.txt
    uv run bevotron tokenizer train --vocab-size 50000       # artifacts/tokenizer/tokenizer_50k.json
    uv run bevotron tokenizer truncate --input artifacts/tokenizer/tokenizer_50k.json --sizes 25000 16000 12000 8000 4000
    uv run bevotron prepare                                  # tokenize train/val -> data/cache/ (per tokenizer)

## Train the baseline

    uv run bevotron info                                     # shape, params, steps, memory
    uv run bevotron run configs/baseline.yaml --ablation baseline
    uv run bevotron sample <run-id> --prompt "Once upon a time"

`--resume` continues an interrupted run from `runs/<id>/ckpt.pt`. Every run writes
`runs/<id>/`: `config.yaml`, `run.json` (git SHA, params, memory), `metrics.jsonl`,
`ckpt.pt` (latest), `ckpt_best.pt` (lowest val loss), `eval.json`.

## Run a study

Every setting lives in `configs/baseline.yaml`. A study is a file in `configs/ablations/`
listing only what differs, using dotted keys. For example, for MLP ratio:

    # configs/ablations/mlp_ratio.yaml
    name: mlp_ratio
    question: How thin can the MLP be?
    grid:
      model.mlp_ratio: [2, 3]

- `grid`: each key's values are crossed (two keys x three values = nine runs)
- `pairs`: explicit combinations, one run each (e.g. a width with its matching head count)
- `overrides`: applied to every run in the file

The baseline is always run 0, so every study compares against it. Then:

    uv run bevotron sweep configs/ablations/mlp_ratio.yaml           # list the runs (dry run)
    uv run bevotron sweep configs/ablations/mlp_ratio.yaml --write   # create runs/<id>/config.yaml
    uv run bevotron run runs/<id>/config.yaml --ablation mlp_ratio   # once per run id
    uv run bevotron judge <id>                                       # optional: LLM-judged story quality
    uv run bevotron compare mlp_ratio                                # results/mlp_ratio.csv vs the baseline

To run every run of a study:

    for d in $(grep -l '"ablation": "mlp_ratio"' runs/*/run.json | xargs -n1 dirname); do
      uv run bevotron run $d/config.yaml --ablation mlp_ratio
    done

Run IDs are a hash of the full config, so identical configs share a run and changing the
baseline gives every study fresh IDs. For a one-off experiment, copy `configs/baseline.yaml`,
edit it, and `bevotron run` it directly. Constraints: `n_head` must divide `d_model` and be a
multiple of `n_kv_head` (use `pairs` to change both together).

## Scores

- **Perplexity** on the full validation split: `perplexity_fp` (full precision) and
  `perplexity` (under the run's quantization scheme). Ranks every run.
- **LLM judge** (`uv run bevotron judge <id>`, needs `uv sync --extra judge` and
  `ANTHROPIC_API_KEY`): the model continues 100 validation-story openings; a judge grades
  grammar, creativity and consistency (1-10). Stories and scores land in `runs/<id>/samples.jsonl`. 
  Default judge is Sonnet 4.6.
- **Bits per byte**: the same loss normalized by bytes of text; use it for the vocab study,
  where per-token perplexity is not comparable across tokenizers.
- **Memory**: `memory_mb` is non-embedding weight + KV-cache bytes at the run's precision
  and context length, so it does not change with vocabulary. `embedding_mb` is the
  embeddings as the evaluated model stores them: the output-layer table (tied, also the
  input table) at the weight precision, and learned positions or an untied input table at
  fp32, since PTQ and QAT round only `nn.Linear` weights. `memory_with_embedding_mb` adds
  it: the size a chip needs if it runs the output layer, which reads the whole table for
  every token. `embedding_params` is listed apart from `total_params` for the same reason.

`bevotron compare <study>` writes all of these plus deltas vs the baseline to `results/<study>.csv`.

## Quantization (`configs/ablations/precision.yaml`: W4A4KV4 and W4A8KV4, PTQ and QAT, dynamic and static scales)

Train the baseline first. The PTQ rows share its training key, so `bevotron run` reuses the
checkpoint and only evaluates; the QAT row loads that checkpoint, turns on fake quantization
with a straight-through estimator, and fine-tunes for `quant.qat_tokens` at `quant.qat_lr`.
Weights get one scale per output channel, activations and the KV cache one scale per token
(per head). `bevotron sample <id> --quant` shows what the scheme does to the text.

`quant.scales: static` swaps the per-token scales for one fixed scale per Linear input and per
KV head, which an integer-only chip needs (no per-token max, and requantization folds into an
integer multiply and shift). PTQ rows calibrate those scales at the 99.99th percentile of 128
training windows; QAT rows start there and learn them (log scales, lr x 10, no decay), and the
checkpoint carries them as extra `*_log_scale` entries. Left at `dynamic`, the key stays out
of the config dict, so run IDs from before static scales existed are unchanged.

A run ID hashes the whole config, so a study only lands on an already-trained baseline if it
expands against that run's saved config. The trained baseline used the 16K-batch laptop
settings, so the precision study is expanded with
`--baseline runs/<baseline-id>/config.yaml` (see the comment in the study file).

## Integer-only inference (`bevotron int`)

`quant/int_engine.py` runs bevo with integers only: int4/int8 codes, int32 matmul
accumulators, every rescale an integer multiply plus rounding shift, LayerNorm with integer
mean, variance, square root and division, GELU and exp as lookup tables, softmax by integer
division, and integer logits (greedy decoding is an integer argmax). `quant/int_export.py`
is the offline step that folds the static scales (learned by a static-QAT run, or calibrated)
into those integers. A test runs the forward pass and generation under a guard that fails on
any floating-point op.

    uv run bevotron int eval <run_id> --windows 128             # loss on validation windows
    uv run bevotron int eval <run_id> --a-bits 8 --acc-bits 16  # sweep any bit width
    uv run bevotron int generate <run_id> --prompt "Once upon a time"
    uv run bevotron int export <run_id> --out model.pt          # the integer model itself

Bit widths default to the run's scheme: `--w-bits --a-bits --kv-bits`, `--out-bits` (q, o,
up and down outputs, 8), `--acc-bits` (32, saturating below), `--p-bits` (attention
probabilities, 8), `--res-bits` (residual stream, 16), `--logit-bits` (16), `--pos-bits` (8)
and `--lut-res --lut-size` (exp table). Each eval appends a row to `results/int_engine.jsonl`
with the largest value every matmul accumulator reached. Matmuls run exactly on CUDA int8
tensor cores (`torch._int_mm`) or CPU int32, about 0.3 s per 512-token window on a laptop GPU.

### Sampled scales (a mix of dynamic and static)

`quant.scales: first_row | first_n | row_first_n` gives each quantized matrix (a Linear's
input, or one head's keys or values) a scale taken from part of itself: the max of row 0,
of the matrix's first `quant.scale_n` values, or of each row's own first N values. That
max is multiplied by a headroom factor (`quant.scale_margin`), which QAT learns per Linear
input and per KV head. On a chip only the sampled values wait at int16, and every other
value goes straight to int4. `bevotron int` runs the same modes with integers only
(`--scales --scale-n --scale-margin --mid-bits`; default: the run's own mode).

## Vocabulary (`configs/ablations/vocab_size.yaml`)

Every run trains from scratch, so run this study on a data-center GPU with the 131K-batch
baseline. Compare runs by bits per byte, not perplexity. The study holds the amount of text
fixed rather than the token count: a smaller vocabulary spends more tokens per story, so each
row's `train.tokens` is the baseline budget scaled by that tokenizer's train-split token count
(the numbers and the formula are in the study file; regenerate them after retraining the
tokenizer or changing the baseline budget).

## Comparison site (`bevotron site`)

A page on http://localhost:8642 for weighing every ablation's quality against its hardware
cost. It only shows saved data, so it needs no GPU or extra packages and runs no model. It
only listens on 127.0.0.1.

    uv run bevotron registry                 # rebuild configs/models.json from ./runs
    uv run bevotron site                     # --port to change 8642

- **Overview**: quality (BPB, perplexity or judge score) against hardware cost (ms/token at
  a chosen link bandwidth, MB streamed per token, model MB or MACs/token), with each
  study's Pareto frontier. The table lists every model's change from its study's baseline,
  scores, embedding and total params, model MB, ms/token and speed against the baseline.
- **Compare**: the picked models side by side, every metric in one row with the best value
  marked, and each model's saved samples from `results/report.md` below.

`configs/models.json` lists every run: its study, the change it makes, its scores, its
saved samples, its config and its folder on the Hugging Face Hub. `weights` names the run
whose folder holds the checkpoint, because PTQ runs share their baseline's weights. Names
and notes edited by hand survive `bevotron registry`. To publish runs for the rest of the
team:

    uv sync --extra train --extra hub        # torch, huggingface_hub, safetensors
    hf auth login                            # a token with write access to the org
    uv run bevotron hub push [ids...] --repo LonghornSilicon/bevo-ablations

Each run uploads to `<repo>/<run id>/`: config.yaml, run.json, eval.json, tokenizer.json,
and model.safetensors for runs that own a checkpoint. Weights are saved without optimizer
state, and the tied output layer is dropped because the model re-ties it on load. The push
marks the runs uploaded in configs/models.json, so commit that file afterwards.

Hardware cost follows `eval/latency.py`: Reef streams weights and the KV cache from the host
for every token, so ms/token = bytes streamed / link bandwidth, compute ignored. With
"output layer on chip" ticked, the vocab x d_model output table is streamed too.

## Layout

    configs/baseline.yaml, configs/ablations/   the shared baseline; one file per study
    src/bevotron/config.py, sweep.py, cli.py    typed config, run expansion, CLI
    src/bevotron/models/                        bevo: transformer, attention (MHA/GQA/MQA, window, KV cache), MLP, norms
    src/bevotron/tokenizer/                     byte-level BPE (train once, truncate)
    src/bevotron/data/                          dataset registry; tokenized cache per (dataset, tokenizer)
    src/bevotron/train/                         loop, AdamW, warmup+cosine, checkpoints, resume
    src/bevotron/quant/                         simulated PTQ and QAT: int8, w4a8, w4a4kv4, w4a8kv4
    src/bevotron/eval/                          perplexity, bits per byte, LLM judge, memory, latency floor
    src/bevotron/site/                          comparison site: registry, Hub upload, cost model, server, page
    runs/<id>/                                  gitignored run outputs
    results/<study>.csv                         committed: each run vs the baseline