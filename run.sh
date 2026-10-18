#!/usr/bin/env bash
# One button: set up the environment if needed, then train.
#   bash run.sh                                         # TinyStories-28M
#   bash run.sh config/train_shakespeare_char.py        # Shakespeare sanity run
#   bash run.sh config/train_tinystories_28m.py --n_layer=4   # any override
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
if [[ "${1:-}" == --* || -z "${1:-}" ]]; then
    python -u train.py config/train_tinystories_28m.py "$@"
else
    python -u train.py "$@"
fi
