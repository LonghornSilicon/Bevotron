r"""Byte-level BPE training and encoding. Python 3.10+, standard library only.

Uses the GPT-2 pre-tokenization pattern, implemented with Unicode categories:
    '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
Training uses weighted pre-token counts, an inverted index, and a lazy heap.
"""

from __future__ import annotations

import heapq
import json
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Callable


DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>",)
_WHITESPACE = frozenset("\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")


def _kind(char: str) -> str:
    if char in _WHITESPACE:
        return "space"
    category = unicodedata.category(char)[0]
    return category if category in ("L", "N") else "symbol"


def pretokenize(text: str) -> Iterator[str]:
    """Yield GPT-2-style pieces without dropping spaces or punctuation."""
    i = 0
    while i < len(text):
        if text[i] == "'":
            contraction = next((s for s in ("s", "d", "m", "t", "ll", "ve", "re")
                                if text.startswith(s, i + 1)), None)
            if contraction is not None:
                end = i + 1 + len(contraction)
                yield text[i:end]
                i = end
                continue
        start = i
        j = i + 1 if text[i] == " " and i + 1 < len(text) else i
        kind = _kind(text[j])
        if kind != "space":
            end = j + 1
            while end < len(text) and _kind(text[end]) == kind:
                end += 1
        else:
            end = i + 1
            while end < len(text) and _kind(text[end]) == "space":
                end += 1
            # The negative-lookahead whitespace alternative leaves the last
            # space available to attach to a following letter/number/symbol run.
            if end < len(text) and end - i > 1:
                end -= 1
        yield text[start:end]
        i = end


def _special_pattern(special_tokens: Iterable[str]):
    tokens = tuple(dict.fromkeys(special_tokens))
    if any(not token for token in tokens):
        raise ValueError("Special tokens must not be empty.")
    if not tokens:
        return None
    return re.compile("|".join(re.escape(t) for t in sorted(tokens, key=lambda t: (-len(t), t))))


def _segments(path: str | Path, special_tokens: Iterable[str], chunk_size: int = 1 << 20):
    """Stream text between special tokens, preserving UTF-8 and line endings.

Memory includes the longest segment between special tokens. With no special
tokens, this necessarily buffers the entire file to preserve pre-tokenization.
"""
    tokens = tuple(special_tokens)
    pattern = _special_pattern(tokens)
    longest = max(map(len, tokens), default=0)
    buffer = ""
    with open(path, "r", encoding="utf-8", newline="") as source:
        while True:
            chunk = source.read(chunk_size)
            buffer += chunk
            if not chunk:
                yield from pattern.split(buffer) if pattern else (buffer,)
                break
            if pattern is None:
                continue
            start = 0
            for match in pattern.finditer(buffer):
                # Delay matches near the chunk edge: a longer special token
                # with the same prefix might continue into the next chunk.
                if match.start() > len(buffer) - longest:
                    break
                yield buffer[start:match.start()]
                start = match.end()
            buffer = buffer[start:]


def _pairs(word):
    return Counter(zip(word, word[1:]))


def _merge(word, pair, new_id):
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


@dataclass(frozen=True)
class _Candidate:
    count: int
    byte_pair: tuple[bytes, bytes]
    pair: tuple[int, int]

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        # Match CS336's lexicographically greatest byte-pair tie-break.
        return self.byte_pair > other.byte_pair


def train_bpe(
    input_path: str | Path,
    vocab_size: int = 10000,
    special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    *,
    max_segments: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Learn a vocabulary and ordered merges from a UTF-8 text file.

    Special tokens reserve vocabulary IDs and delimit training segments, but
    never contribute to pair counts. max_segments optionally limits training
    to a prefix of document segments for debugging. Vocabulary may finish
    below vocab_size when the corpus has no remaining pairs.
    """
    specials = tuple(dict.fromkeys(special_tokens))
    _special_pattern(specials)
    if vocab_size < 256 + len(specials):
        raise ValueError("Vocabulary must fit all 256 bytes plus special tokens.")
    if max_segments is not None and max_segments < 1:
        raise ValueError("max_segments must be positive.")
    vocab = {i: bytes([i]) for i in range(256)}
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("Special tokens must not duplicate a base-byte token.")
    token_ids = {value: key for key, value in vocab.items()}
    special_bytes = {token.encode("utf-8") for token in specials}
    frequencies = Counter()
    for index, segment in enumerate(_segments(input_path, specials)):
        if max_segments is not None and index >= max_segments:
            break
        frequencies.update(piece.encode("utf-8") for piece in pretokenize(segment))
    if progress:
        progress(f"Counted {sum(frequencies.values()):,} pre-tokens ({len(frequencies):,} unique).")
    words = [tuple(piece) for piece in frequencies]
    weights = list(frequencies.values())
    del frequencies
    counts = Counter()
    locations = defaultdict(set)
    for index, word in enumerate(words):
        for pair, amount in _pairs(word).items():
            counts[pair] += amount * weights[index]
            locations[pair].add(index)
    heap = [_Candidate(count, (vocab[pair[0]], vocab[pair[1]]), pair)
            for pair, count in counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(vocab) < vocab_size and heap:
        candidate = heapq.heappop(heap)
        pair = candidate.pair
        if candidate.count != counts.get(pair, 0) or candidate.count <= 0:
            continue
        combined = vocab[pair[0]] + vocab[pair[1]]
        if combined in special_bytes:
            raise ValueError("A learned token collided with a special token.")
        new_id = token_ids.get(combined)
        if new_id is None:
            new_id = len(vocab)
            vocab[new_id] = combined
            token_ids[combined] = new_id
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        deltas = Counter()
        for index in list(locations.get(pair, ())):
            before = _pairs(words[index])
            words[index] = _merge(words[index], pair, new_id)
            after = _pairs(words[index])
            for changed_pair in before.keys() | after.keys():
                deltas[changed_pair] += (after[changed_pair] - before[changed_pair]) * weights[index]
                if after[changed_pair]:
                    locations[changed_pair].add(index)
                else:
                    locations[changed_pair].discard(index)
                    if not locations[changed_pair]:
                        del locations[changed_pair]
        for changed_pair, delta in deltas.items():
            if not delta:
                continue
            counts[changed_pair] += delta
            count = counts[changed_pair]
            if count:
                heapq.heappush(heap, _Candidate(count, (vocab[changed_pair[0]], vocab[changed_pair[1]]), changed_pair))
            else:
                del counts[changed_pair]
        # Bound stale heap entries rather than retaining every past update.
        if len(heap) > max(1024, 4 * len(counts)):
            heap = [_Candidate(n, (vocab[p[0]], vocab[p[1]]), p) for p, n in counts.items()]
            heapq.heapify(heap)
        if progress and (len(merges) == 1 or len(merges) % 100 == 0):
            progress(f"Vocabulary: {len(vocab):,}/{vocab_size:,}; merges: {len(merges):,}")
    return vocab, merges


class BPETokenizer:
    """Encode/decode using a byte vocabulary and ordered BPE merge rules."""

    def __init__(self, vocab, merges, special_tokens=DEFAULT_SPECIAL_TOKENS):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = tuple(dict.fromkeys(special_tokens))
        self._special_pattern = _special_pattern(self.special_tokens)
        if set(self.vocab) != set(range(len(self.vocab))):
            raise ValueError("Vocabulary IDs must be contiguous from zero.")
        if any(self.vocab.get(i) != bytes([i]) for i in range(256)):
            raise ValueError("The first 256 vocabulary entries must be raw bytes.")
        self._ids = {value: key for key, value in self.vocab.items()}
        if len(self._ids) != len(self.vocab):
            raise ValueError("Vocabulary byte sequences must be unique.")
        for token in self.special_tokens:
            if token.encode("utf-8") not in self._ids:
                raise ValueError(f"Special token is missing from vocabulary: {token!r}")
        self._ranks = {}
        available = {bytes([i]) for i in range(256)}
        for rank, pair in enumerate(self.merges):
            if pair in self._ranks or any(part not in available for part in pair):
                raise ValueError("Invalid or duplicate merge rule.")
            if pair[0] + pair[1] not in self._ids:
                raise ValueError("Merged token is missing from vocabulary.")
            self._ranks[pair] = rank
            available.add(pair[0] + pair[1])
        self._cached_encode = lru_cache(maxsize=65536)(self._encode_piece)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eos_token_id(self):
        return self._ids.get(b"<|endoftext|>")

    def _encode_piece(self, piece):
        parts = tuple(bytes([byte]) for byte in piece)
        while len(parts) > 1:
            candidates = ((self._ranks[pair], pair) for pair in zip(parts, parts[1:])
                          if pair in self._ranks)
            best = min(candidates, default=None)
            if best is None:
                break
            pair = best[1]
            parts = _merge(parts, pair, pair[0] + pair[1])
        return tuple(self._ids[part] for part in parts)

    def encode(self, text: str) -> list[int]:
        result = []
        start = 0
        matches = self._special_pattern.finditer(text) if self._special_pattern else ()
        for match in matches:
            for piece in pretokenize(text[start:match.start()]):
                result.extend(self._cached_encode(piece.encode("utf-8")))
            result.append(self._ids[match.group().encode("utf-8")])
            start = match.end()
        for piece in pretokenize(text[start:]):
            result.extend(self._cached_encode(piece.encode("utf-8")))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode each supplied string independently and yield IDs lazily.

        Supply complete documents/records, not arbitrary character chunks.
        Chunk boundaries act as pre-token boundaries and can affect token IDs.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Join bytes before UTF-8 decoding; generated invalid bytes use errors."""
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors=errors)

    def truncate(self, vocab_size: int) -> "BPETokenizer":
        """Return a smaller tokenizer built from a prefix of this one's merges.

        BPE learns merges sequentially, choosing the most frequent current pair.
        With identical training data, preprocessing, special tokens, and tie-breaking,
        a smaller run learns the same initial sequence of merges.
        """
        base = 256 + len(self.special_tokens)
        if not base <= vocab_size <= self.vocab_size:
            raise ValueError(f"vocab_size must be between {base} and {self.vocab_size}.")
        vocab = {i: bytes([i]) for i in range(256)}
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")
        ids = {value: key for key, value in vocab.items()}
        merges = []
        for pair in self.merges:
            if len(vocab) >= vocab_size:
                break
            merges.append(pair)
            combined = pair[0] + pair[1]
            if combined not in ids:
                ids[combined] = len(vocab)
                vocab[len(vocab)] = combined
        return BPETokenizer(vocab, merges, self.special_tokens)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "bevotron-byte-bpe-v1",
            "pretokenizer": "gpt2-unicode-v1",
            "unicode_version": unicodedata.unidata_version,
            "vocab": [self.vocab[i].hex() for i in range(len(self.vocab))],
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "bevotron-byte-bpe-v1" or data.get("pretokenizer") != "gpt2-unicode-v1":
            raise ValueError("Unsupported tokenizer format.")
        if data["unicode_version"] != unicodedata.unidata_version:
            # Unicode database versions can classify some characters differently,
            # potentially changing pre-tokenization. Warn to allow loading, but use
            # matching Python/Unicode versions for reproducible experiments.
            warnings.warn(
                f"Tokenizer was trained with Unicode {data['unicode_version']}; this Python "
                f"has {unicodedata.unidata_version}. Pre-tokenization of rare characters "
                "could differ.", RuntimeWarning, stacklevel=2)
        vocab = {i: bytes.fromhex(value) for i, value in enumerate(data["vocab"])}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
        return cls(vocab, merges, data["special_tokens"])
