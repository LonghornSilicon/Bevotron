#!/usr/bin/env bash
# One button: set up the environment if needed, then train.
#   bash run.sh                                   # TinyStories-28M (GPT-Neo) from scratch
#   bash run.sh config/rope.py                    # an experiment, saved to out/rope
#   bash run.sh config/rope.py --max_iters=2000   # override any setting at the top of train.py
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python -u train.py "$@"
