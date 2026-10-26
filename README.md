# Bevotron

Longhorn Silicon's small-scale LLM training pipeline.

## Run

    bash run.sh                                    # TinyStories-28M (GPT-Neo, 52.0M params) from scratch
    bash run.sh --n_layer=4 --max_iters=2000       # override any setting
    bash run.sh --init_from=roneneldan/TinyStories-28M --eval_only=True   # score the published weights

First run creates `.venv`, installs requirements, and prepares the data.
Checkpoints go to `out-*/`. Every setting lives at the top of `train.py`.

`model.py` is GPT-Neo in one file, the architecture of the TinyStories checkpoints;
`GPT.from_pretrained` loads any of them from the Hugging Face hub.
Training loop and config system from Karpathy's nanoGPT.
