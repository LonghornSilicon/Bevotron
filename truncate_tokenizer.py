"""Derive smaller tokenizers from one trained tokenizer by truncating its merges.

Example (train once at 50K, then):
    python truncate_tokenizer.py --input artifacts/tokenizer/tokenizer_50k.json \
        --sizes 25000 16000 12000 8000 4000 --output-dir artifacts/tokenizer
"""

import argparse
from pathlib import Path

from tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Trained tokenizer JSON")
    parser.add_argument("--sizes", type=int, nargs="+", required=True, help="Target vocab sizes")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer"))
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.input)
    for size in args.sizes:
        small = tokenizer.truncate(size)
        out = args.output_dir / f"tokenizer_{size}.json"
        small.save(out)
        print(f"{out}: {small.vocab_size:,} tokens, {len(small.merges):,} merges")


if __name__ == "__main__":
    main()
