"""Tokenize TinyStories with the GPT-Neo tokenizer into train.bin / val.bin.

Same format as nanoGPT's data files: one flat uint16 array of token ids,
stories separated by the end-of-text token (50256). Run once (~10-30 min).
"""

import os
import pickle

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

out_dir = os.path.dirname(os.path.abspath(__file__))
num_proc = max(1, (os.cpu_count() or 2) // 2)

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # TinyStories' tokenizer
eot = tok.eos_token_id  # 50256


def encode(batch):
    ids = tok(batch["text"])["input_ids"]
    ids = [x + [eot] for x in ids]
    return {"ids": ids, "len": [len(x) for x in ids]}


ds = load_dataset("roneneldan/TinyStories")
for split, name in [("train", "train"), ("validation", "val")]:
    d = ds[split].map(encode, batched=True, remove_columns=ds[split].column_names,
                      num_proc=num_proc, desc=f"tokenizing {split}")
    total = int(np.sum(d["len"], dtype=np.uint64))
    path = os.path.join(out_dir, f"{name}.bin")
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
    shards = min(1024, len(d))
    i = 0
    for s in tqdm(range(shards), desc=f"writing {path}"):
        chunk = d.shard(num_shards=shards, index=s, contiguous=True).with_format("numpy")
        flat = np.concatenate(chunk["ids"])
        arr[i:i + len(flat)] = flat
        i += len(flat)
    arr.flush()
    print(f"{name}: {total:,} tokens")

with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": tok.vocab_size}, f)  # 50257, read by train.py
print(f"vocab_size = {tok.vocab_size}")