# Bevotron

Longhorn Silicon's small-scale LLM training pipeline.

## Run

    bash run.sh                                    # TinyStories-28M (GPT-Neo, 52.0M params) from scratch
    bash run.sh config/rope.py                     # an experiment, saved to out/rope
    bash run.sh config/rope.py --max_iters=2000    # override any setting
    bash run.sh --init_from=roneneldan/TinyStories-28M --eval_only=True   # score the published weights

First run creates `.venv`, installs requirements, and prepares the data.
