# Bevotron

Longhorn Silicon's small-scale LLM training pipeline for pre-training and post-training.

## Run

    bash run.sh                                    # TinyStories-28M (GPT-Neo, 52.0M params) from scratch
    bash run.sh config/train_shakespeare_char.py   # 5-minute sanity run
    bash run.sh --n_layer=4 --max_iters=2000       # override any setting

First run creates `.venv`, installs requirements, and prepares the data.
Checkpoints go to `out-*/`. Every setting lives in `config/*.py`.

Training loop and config system from Karpathy's nanoGPT (MIT, `LICENSE-nanoGPT`);
GPT-Neo model via `model_neo.py` over Hugging Face's implementation.
