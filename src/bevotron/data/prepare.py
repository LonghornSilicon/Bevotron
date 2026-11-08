"""Tokenize a dataset once per tokenizer into data/cache/<dataset>/<tokenizer>/{train,val}.bin.

The cache key is the tokenizer file's content hash, so retraining the tokenizer
invalidates the cache and truncated vocabularies get their own entries. meta.json
records the token and byte counts the bits-per-byte score needs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bevotron.config import RunConfig
from bevotron.data.registry import DATASETS
from bevotron.tokenizer.bpe import BPETokenizer

CACHE_ROOT = Path("data/cache")
TOKENIZER_ROOT = Path("artifacts/tokenizer")


@dataclass
class DatasetMeta:
    dataset: str
    tokenizer_path: str
    tokenizer_hash: str
    vocab_size: int
    eos_token_id: int
    dtype: str
    train_tokens: int
    val_tokens: int
    val_bytes: int  # UTF-8 bytes of the raw validation text (B in bits-per-byte)


def resolve_tokenizer_path(config: RunConfig) -> Path:
    if config.tokenizer.path:
        return Path(config.tokenizer.path)
    return TOKENIZER_ROOT / f"tokenizer_{config.tokenizer.vocab_size // 1000}k.json"


def _tokenizer_hash(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `bevotron tokenizer train` (and `truncate`) first"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def cache_dir(config: RunConfig) -> Path:
    return CACHE_ROOT / config.data.dataset / _tokenizer_hash(resolve_tokenizer_path(config))


_TOK: BPETokenizer | None = None


def _init_worker(path: str) -> None:
    global _TOK
    _TOK = BPETokenizer.load(path)


def _encode_batch(texts: list[str]) -> list[int]:
    out: list[int] = []
    eos = _TOK.eos_token_id
    for text in texts:
        out.extend(_TOK.encode(text))
        out.append(eos)
    return out


def _encode_split(
    split,
    text_field: str,
    tokenizer_path: Path,
    dtype,
    out_path: Path,
    workers: int,
    batch: int = 512,
) -> tuple[int, int]:
    """Stream a split through the tokenizer; returns (tokens, bytes)."""
    texts = (row[text_field] for row in split)
    batches = []
    current: list[str] = []
    n_bytes = 0
    for text in texts:
        n_bytes += len(text.encode("utf-8"))
        current.append(text)
        if len(current) == batch:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    n_tokens = 0
    with open(out_path, "wb") as f, Pool(workers, _init_worker, (str(tokenizer_path),)) as pool:
        for ids in pool.imap(_encode_batch, batches, chunksize=4):
            np.asarray(ids, dtype=dtype).tofile(f)
            n_tokens += len(ids)
    return n_tokens, n_bytes


def prepare(config: RunConfig, workers: int = 8, force: bool = False) -> DatasetMeta:
    """Return the cached dataset for this config, tokenizing it first if needed."""
    folder = cache_dir(config)
    meta_path = folder / "meta.json"
    if meta_path.exists() and not force:
        return DatasetMeta(**json.loads(meta_path.read_text()))
    from datasets import load_dataset  # heavy import, only when actually preparing

    spec = DATASETS[config.data.dataset]
    tokenizer_path = resolve_tokenizer_path(config)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"{tokenizer_path} not found; run `bevotron tokenizer train` / `truncate` first"
        )
    tok = BPETokenizer.load(tokenizer_path)
    if tok.vocab_size != config.tokenizer.vocab_size:
        raise ValueError(
            f"tokenizer has {tok.vocab_size} tokens, config says {config.tokenizer.vocab_size}"
        )
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    folder.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(spec.hf_name)
    train_tokens, _ = _encode_split(
        ds[spec.train_split], spec.text_field, tokenizer_path, dtype, folder / "train.bin", workers
    )
    val_tokens, val_bytes = _encode_split(
        ds[spec.val_split], spec.text_field, tokenizer_path, dtype, folder / "val.bin", workers
    )
    meta = DatasetMeta(
        config.data.dataset,
        str(tokenizer_path),
        _tokenizer_hash(tokenizer_path),
        tok.vocab_size,
        tok.eos_token_id,
        np.dtype(dtype).name,
        train_tokens,
        val_tokens,
        val_bytes,
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2))
    return meta


def load_split(config: RunConfig, split: str) -> np.memmap:
    meta = prepare(config)
    return np.memmap(cache_dir(config) / f"{split}.bin", dtype=np.dtype(meta.dtype), mode="r")


def export_text(
    dataset: str, split: str, out_path: Path, limit: int | None = None
) -> tuple[Path, int]:
    """Write a split as plain UTF-8 text, documents separated by <|endoftext|>, for tokenizer
    training. Uses the same source as prepare(), so the tokenizer sees the training data."""
    from datasets import load_dataset

    spec = DATASETS[dataset]
    rows = load_dataset(
        spec.hf_name, split=spec.train_split if split == "train" else spec.val_split
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row[spec.text_field])
            f.write("<|endoftext|>")
            n += 1
            if limit is not None and n >= limit:
                break
    return out_path, n
