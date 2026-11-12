# Static quantization scales for bevo

**Bottom line:** W4A4KV4 with static scales, trained with QAT and run fully integer with
int16 accumulators, reaches 3.98 perplexity against 3.00 at full precision, and its stories
stay coherent. W4A8KV4 static QAT is almost lossless (3.10).

## Setup

- **Model:** bevo, 8 layers, width 512, 25.2M non-embedding parameters, 50k vocabulary,
  TinyStories, 504M training tokens (baseline `27cb45751c`, 131K-token batch). 14.7 MB at
  4 bits.
- **Static scales:** one fixed scale per Linear input and one per KV-cache head, so no scale
  is computed at inference.
  - **Calibration:** each scale starts at the 99.99th percentile of absolute values over 128
    training windows (8 batches of 16, seed 0).
  - **QAT:** 50M tokens of fine-tuning from the baseline. The scales are learned along with
    the weights (stored as log scales, LSQ gradient, 10x learning rate, no weight decay).
- **Weights:** 4 bits, one scale per output channel.
- **Code:** `quant/static.py` (scales), `quant/int_engine.py` and `quant/int_export.py`
  (integer engine), on branch `feature/static-scales` and later.

## Accuracy (B300, full validation split, float simulation)

| Run | Scheme | Method | Scales | Val loss | Perplexity |
|---|---|---|---|---|---|
| `27cb45751c` | int8 (reference) | PTQ | per-token | 1.385 | 4.00 |
| `ee390af080` | W4A8KV4 | PTQ | per-token | 1.459 | 4.30 |
| `98aa072451` | W4A8KV4 | PTQ | **static** | 1.490 | 4.44 |
| `844909e3ec` | W4A8KV4 | QAT | **static** | **1.411** | **4.10** |
| `714841d60b` | W4A4KV4 | PTQ | per-token | 3.333 | 28.0 |
| `f77ecaab23` | W4A4KV4 | PTQ | **static** | 5.416 | 225 |
| `a5ad4528c4` | W4A4KV4 | QAT | per-token | 1.549 | 4.70 |
| `cd276db15a` | W4A4KV4 | QAT | **static** | **1.667** | **5.30** |

- **4-bit activations need QAT.** Without it, static W4A4KV4 is unusable (225). With it, the
  gap to per-token scales shrinks to 0.12 in loss.
- **With 8-bit activations, static QAT is better than per-token PTQ** (4.10 vs 4.30).
- **QAT is cheap:** 381 steps, about 3 minutes on one B300 (132 s per-token, 184 s and
  181 s static).

## Fully integer inference (first 128 validation windows)

| Run | Scheme | Full precision | Float simulation | **Integer engine** |
|---|---|---|---|---|
| `27cb45751c` | int8 (reference) | 3.00 | 3.01 | n/a |
| `844909e3ec` | W4A8KV4 static QAT | 3.05 | 3.08 | **3.10** |
| `cd276db15a` | W4A4KV4 static QAT | 3.42 | 3.97 | **3.98** |
| `a5ad4528c4` | W4A4KV4 per-token QAT, calibrated static scales | 3.28 | 3.52 | 4.32 |

The integer engine matches the float simulation within 0.02, so going fully integer costs
nothing extra. The last row shows that a model trained with per-token scales loses accuracy
when switched to static scales afterwards. The scales need to be learned during QAT.

## The three chip requirements

| Requirement | Status |
|---|---|
| 1. INT8 matmul outputs, no FP accumulation | **Done.** Matmul outputs are int8 codes. An **int8 accumulator fails** (perplexity 223 for W4A4KV4). **int16 is enough for W4A4KV4:** with `--acc-bits 16` the result is identical (3.98, loss 1.3813), and the largest accumulator value is 9,280, under the 32,767 limit. |
| 2. Static scaling, not dynamic | **Done.** Calibrated, then learned in QAT (above). |
| 3. Integer-only requantization, no FP anywhere | **Done.** Every rescale is an integer multiply plus a rounding shift. LayerNorm uses integer mean, variance, square root and division. Softmax and GELU use lookup tables and integer division. Logits and sampling are integer. `tests/test_int_engine.py` fails if any floating-point operation runs. |

Accumulator peaks for W4A4KV4 static QAT (`cd276db15a`):

| Matmul | Largest value |
|---|---|
| Attention scores | 9,280 |
| Probabilities x V | 2,056 |
| k | 1,347 |
| Output layer | 1,168 |
| q | 1,123 |
| v | 1,080 |
| down | 1,075 |
| o | 976 |
| up | 536 |

## Follow-up: mixed dynamic+static scales

Each matrix takes its scale from part of itself, times a headroom factor learned in QAT
(`quant/hybrid.py`, branch `feature/hybrid-scales`). All rows are W4A4KV4 QAT, run fully
integer with int16 accumulators on the first 128 validation windows.

| Run | Scale from | Integer perplexity | Full-split val loss | Largest accumulator |
|---|---|---|---|---|
| `cd276db15a` | Static (above) | 3.98 | 1.667 | 9,280 |
| `2d7a069f56` | First row | 3.68 | 1.606 | 17,003 |
| `ee2f2ec9f2` | First 8 values of the matrix | 3.71 | 1.591 | 13,599 |
| `9a05eb501f` | First 16 values of the matrix | 3.72 | 1.591 | 15,606 |
| `a717e92627` | First 64 values of the matrix | 3.73 | 1.601 | 19,183 |
| `65d98719b6` | First 128 values of the matrix | 3.63 | 1.573 | 17,480 |
| `64917039df` | First 8x8 tile | 3.62 | 1.567 | 15,729 |
| `b918baa448` | First 16x16 tile | 3.48 | 1.531 | 19,792 |
| `d33add49d3` | Each row's first 8 values | 3.76 | 1.595 | 18,971 |
| `634e1145be` | Each row's first 16 values | 3.66 | 1.580 | 17,344 |
| `38099ca695` | Each row's first 64 values | 3.47 | 1.533 | 24,546 |
| `9e0f9f50a4` | Each row's first 128 values | 3.44 | 1.520 | 28,168 |

For comparison, per-token dynamic scales (`a5ad4528c4`) give 3.52 in float simulation and a
full-split loss of 1.549. Every mixed variant beats static. The cost is an int16 holding area
for the sampled values and one integer divide per value. The 16x16 tile is the best option
that keeps one scale per matrix.

## Reproduce

    uv run bevotron sweep configs/ablations/precision.yaml --write
    uv run bevotron run runs/<id>/config.yaml                      # train and score
    uv run bevotron int eval <id> --windows 128 --acc-bits 16      # fully integer
    uv run bevotron int generate <id> --prompt "Once upon a time"

## Caveats

- One model size, one seed, one dataset.
- The 128-window numbers compare methods with each other. The full-split numbers are the
  ones to quote.
- Story quality has no judge scores yet (the LLM judge needs `ANTHROPIC_API_KEY`).
- Checkpoints live on one laptop, not in the repo.
