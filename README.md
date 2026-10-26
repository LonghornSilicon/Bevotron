# Bevotron

Longhorn Silicon's small-scale LLM training pipeline for pre-training and post-training.

One shared baseline config, one small file per study, one command to turn a study into runs.

## Setup 

    uv sync --extra train
    uv run pytest                                            # 52 tests to make sure everything works
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
- **Memory**: weight + KV-cache bytes at the run's precision and context length.

`bevotron compare <study>` writes all of these plus deltas vs the baseline to `results/<study>.csv`.

## Quantization (`configs/ablations/precision.yaml`, W4A4KV4)

Train the baseline first. The PTQ row shares its training key, so `bevotron run` reuses the
checkpoint and only evaluates; the QAT row loads that checkpoint, turns on fake quantization
with a straight-through estimator, and fine-tunes for `quant.qat_tokens` at `quant.qat_lr`.
`bevotron sample <id> --quant` shows what the scheme does to the text.

## Layout

    configs/baseline.yaml, configs/ablations/   the shared baseline; one file per study
    src/bevotron/config.py, sweep.py, cli.py    typed config, run expansion, CLI
    src/bevotron/models/                        bevo: transformer, attention (MHA/GQA/MQA, window, KV cache), MLP, norms
    src/bevotron/tokenizer/                     byte-level BPE (train once, truncate)
    src/bevotron/data/                          dataset registry; tokenized cache per (dataset, tokenizer)
    src/bevotron/train/                         loop, AdamW, warmup+cosine, checkpoints, resume
    src/bevotron/quant/                         simulated PTQ and QAT: int8, w4a8, w4a4kv4, w4a8kv4
    src/bevotron/eval/                          perplexity, bits per byte, LLM judge, memory, latency floor
    runs/<id>/                                  gitignored run outputs
    results/<study>.csv                         committed: each run vs the baseline