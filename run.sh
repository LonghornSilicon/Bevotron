#!/usr/bin/env bash
# One button: set up the environment if needed, then train.
#   bash run.sh                                   # TinyStories-28M (GPT-Neo) from scratch
#   bash run.sh --n_layer=4 --max_iters=2000      # override any setting at the top of train.py
#   bash run.sh --init_from=roneneldan/TinyStories-28M --eval_only=True   # score the published weights
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python -u train.py "$@"
