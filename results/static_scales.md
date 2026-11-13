# Static quantization scales for bevo

**Bottom line:** W4A4KV4 with static scales, trained with QAT and run fully integer with
int16 accumulators, reaches 3.98 perplexity against 3.00 at full precision, and its stories
stay coherent. 

## Setup

- **Model:** bevo, 8 layers, width 512, 25.2M non-embedding parameters, 50k vocabulary,
  TinyStories, 504M training tokens (baseline `27cb45751c`, 131K-token batch). 14.7 MB at
  4 bits.
- **Static scales:** one fixed scale per Linear input and one per KV-cache head, so no scale
  is computed at inference.
  - **Calibration:** at each spot, take the 99.99th percentile of |x| over 128 training windows (8 batches of 16, seed 0). The scale is that value / 7 for 4-bit (/ 127 for 8-bit), so anything above the percentile gets clipped.
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

## Mixed dynamic+static scales

Each matrix takes its scale from part of itself, times a headroom factor learned in QAT
(`quant/hybrid.py`, branch `feature/hybrid-scales`). All rows are W4A4KV4 QAT, on the first
128 validation windows.

Whole-matrix modes (first row, first N values of the matrix, first tile) set one scale per
matrix from the first rows of the sequence. Text generation runs one token at a time, so the
model keeps that scale from the prompt's first rows and reuses it for every later token
(`ScaleMemory` in `quant/hybrid.py`, carried in the integer engine's cache too). Rows that
arrive before an N x N tile is full add to its max. Per-row modes need no memory. The
generation column decodes every window token by token after a 16-token prompt, with the KV
cache, and scores every prediction.

| Run | Scale from | Integer, whole sequence | **Float, generation** | Full-split val loss | Largest accumulator |
|---|---|---|---|---|---|
| `cd276db15a` | Static (above) | 3.98 | **3.97** | 1.667 | 9,280 |
| `2d7a069f56` | First row | 3.68 | **3.74** | 1.606 | 17,003 |
| `ee2f2ec9f2` | First 8 values of the matrix | 3.71 | **3.69** | 1.591 | 13,599 |
| `9a05eb501f` | First 16 values of the matrix | 3.72 | **3.68** | 1.591 | 15,606 |
| `a717e92627` | First 64 values of the matrix | 3.73 | **3.73** | 1.601 | 19,183 |
| `65d98719b6` | First 128 values of the matrix | 3.63 | **3.63** | 1.573 | 17,480 |
| `64917039df` | First 8x8 tile | 3.62 | **3.60** | 1.567 | 15,729 |
| `b918baa448` | First 16x16 tile | 3.48 | **3.47** | 1.531 | 19,792 |
| `d33add49d3` | Each row's first 8 values | 3.76 | **3.72** | 1.595 | 18,971 |
| `634e1145be` | Each row's first 16 values | 3.66 | **3.63** | 1.580 | 17,344 |
| `38099ca695` | Each row's first 64 values | 3.47 | **3.47** | 1.533 | 24,546 |
| `9e0f9f50a4` | Each row's first 128 values | 3.44 | **3.43** | 1.520 | 28,168 |

Integer runs use int16 accumulators. With the scale carried, generation scores the same as a
whole sequence: identical in float64, within 0.01 in float32, and the integer engine's cached
decoding is bit-exact with its whole-sequence forward (`tests/test_int_engine.py`). A 1-token
prompt, where the tile fills in during generation, moves no result by more than 0.01.

For generating text, each row's first 128 values is the best option (3.43, ahead of
per-token dynamic at 3.52). The 16x16 tile is close behind (3.47) with one scale per matrix
instead of one per token, and both tiles beat first-N-of-the-matrix. Every sampled mode
beats static (3.97).

An earlier version of this table gave each generated token a scale from its own row instead
of the first rows' scale. That did not match training and made the tiles and
first-N-of-the-matrix look worse than static (5.51 for the 8x8 tile, 4.30 for the first 8
values).

## Reproduce

    uv run bevotron sweep configs/ablations/precision.yaml --write
    uv run bevotron run runs/<id>/config.yaml                      # train and score
    uv run bevotron int eval <id> --windows 128 --acc-bits 16      # fully integer
    uv run bevotron int generate <id> --prompt "Once upon a time"
