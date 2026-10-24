"""Train a tokenizer on a UTF-8 file; does not train a language model."""

import argparse
import time
from pathlib import Path

from tokenizer import BPETokenizer, train_bpe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/tokenizer_debug.txt"))
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--max-segments", type=int, help="Debug on the first N document segments")
    parser.add_argument("--special-token", action="append", dest="special_tokens",
                        help="Repeat for multiple tokens; default: <|endoftext|>")
    args = parser.parse_args()
    specials = args.special_tokens if args.special_tokens is not None else ["<|endoftext|>"]
    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, specials,
                             max_segments=args.max_segments, progress=print)
    tokenizer = BPETokenizer(vocab, merges, specials)
    tokenizer.save(args.output)
    sample = "Hello, world! A cat found a shiny toy. <|endoftext|>"
    assert tokenizer.decode(tokenizer.encode(sample), errors="strict") == sample
    print(f"Saved {tokenizer.vocab_size:,} tokens and {len(merges):,} merges to {args.output}")
    print(f"End-of-text ID: {tokenizer.eos_token_id}; elapsed: {time.perf_counter() - start:.2f}s")
    if tokenizer.vocab_size < args.vocab_size:
        print("Stopped early: this sample has no remaining adjacent pairs to merge.")


if __name__ == "__main__":
    main()
