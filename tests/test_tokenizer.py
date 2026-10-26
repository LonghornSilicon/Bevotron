import json
import random
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from bevotron.tokenizer.bpe import BPETokenizer, _segments, pretokenize, train_bpe


def reference_train(text, size, specials):
    """Slow reference: recount every pair after every merge."""
    vocab = {i: bytes([i]) for i in range(256)}
    for special in specials:
        vocab[len(vocab)] = special.encode()
    pattern = "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
    segments = re.split(pattern, text) if pattern else [text]
    words = [
        tuple(bytes([b]) for b in piece.encode())
        for segment in segments
        for piece in pretokenize(segment)
    ]
    merges = []
    while len(vocab) < size:
        counts = Counter(pair for word in words for pair in zip(word, word[1:], strict=False))
        if not counts:
            break
        pair = max(counts, key=lambda p: (counts[p], p))
        combined = b"".join(pair)
        if combined not in vocab.values():
            vocab[len(vocab)] = combined
        merges.append(pair)
        changed = []
        for word in words:
            result = []
            i = 0
            while i < len(word):
                if word[i : i + 2] == pair:
                    result.append(combined)
                    i += 2
                else:
                    result.append(word[i])
                    i += 1
            changed.append(tuple(result))
        words = changed
    return vocab, merges


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "corpus.txt"

    def train(self, text, size=300, specials=("<|endoftext|>",)):
        self.path.write_bytes(text.encode())
        vocab, merges = train_bpe(self.path, size, specials)
        return BPETokenizer(vocab, merges, specials)

    def test_pretokenization(self):
        self.assertEqual(list(pretokenize("cat cat car")), ["cat", " cat", " car"])
        self.assertEqual(
            list(pretokenize("I'm  learning!\n\n")), ["I", "'m", " ", " learning", "!", "\n\n"]
        )
        text = "Hello\t  café你好123🙂\r\n A\u0301\x1c"
        self.assertEqual("".join(pretokenize(text)), text)

    def test_first_merge(self):
        tok = self.train("cat cat car", 258)
        self.assertEqual(tok.merges, [(b"c", b"a")])
        self.assertEqual(tok.vocab_size, 258)

    def test_frequency_weighting(self):
        tok = self.train("cat cat cat dog", 258)
        self.assertEqual(tok.merges, [(b"c", b"a")])

    def test_lexical_tie(self):
        tok = self.train("ab cd", 258)
        self.assertEqual(tok.merges[0], (b"c", b"d"))

    def test_overlap(self):
        tok = self.train("aaaaa", 258)
        self.assertEqual(tok.merges, [(b"a", b"a")])
        self.assertEqual(len(tok.encode("aaaaa")), 3)

    def test_unicode_roundtrip(self):
        tok = self.train("Hello café 世界 🙂<|endoftext|>another story")
        for text in [
            "",
            "new unseen words",
            "你好🌎",
            "café\r\n\t A\u0301",
            "<|endoftext|>",
            "a<|endoftext|><|endoftext|>b",
            "\x00\x01\x1f",
        ]:
            self.assertEqual(tok.decode(tok.encode(text), errors="strict"), text)

    def test_special_boundary(self):
        tok = self.train("a<|endoftext|>b")
        self.assertEqual(tok.merges, [])
        self.assertEqual(tok.encode("<|endoftext|>"), [tok.eos_token_id])

    def test_overlapping_specials(self):
        tok = self.train("x<a>y<a>b>z", specials=("<a>", "<a>b>"))
        long_id = next(i for i, b in tok.vocab.items() if b == b"<a>b>")
        self.assertEqual(tok.encode("<a>b>"), [long_id])

    def test_streamed_training_boundaries(self):
        text = "doc1<a>b>doc2<a>doc3<a>b>"
        specials = ("<a>", "<a>b>")
        self.path.write_bytes(text.encode())
        for chunk_size in range(1, 15):
            self.assertEqual(
                list(_segments(self.path, specials, chunk_size)), ["doc1", "doc2", "doc3", ""]
            )

    def test_save_load(self):
        tok = self.train("cat cat car café<|endoftext|>")
        path = Path(self.temp.name) / "tokenizer.json"
        tok.save(path)
        restored = BPETokenizer.load(path)
        self.assertEqual(restored.vocab, tok.vocab)
        self.assertEqual(restored.merges, tok.merges)
        self.assertEqual(restored.encode("cat café unseen🙂"), tok.encode("cat café unseen🙂"))

    def test_empty_and_small_vocab(self):
        tok = self.train("")
        self.assertEqual(tok.vocab_size, 257)
        self.assertEqual(tok.encode("hi"), [104, 105])
        with self.assertRaises(ValueError):
            self.train("abc", 256)

    def test_encode_iterable(self):
        tok = self.train("hello world hello")
        self.assertEqual(
            list(tok.encode_iterable(["hello", " world"])),
            tok.encode("hello") + tok.encode(" world"),
        )

    def test_invalid_generated_utf8(self):
        tok = self.train("")
        self.assertEqual(tok.decode([255]), "\ufffd")
        with self.assertRaises(UnicodeDecodeError):
            tok.decode([255], errors="strict")

    def test_merge_rank_beats_longest_match(self):
        vocab = {i: bytes([i]) for i in range(256)}
        vocab.update({256: b"ab", 257: b"bc"})
        tok = BPETokenizer(vocab, [(b"b", b"c"), (b"a", b"b")], [])
        self.assertEqual(tok.encode("abc"), [97, 257])

    def test_truncate_matches_direct_training(self):
        text = Path(__file__).read_text(encoding="utf-8")  # any real text with many pairs
        big = self.train(text, 420)
        for size in (300, 350, 400):
            direct = self.train(text, size)
            small = big.truncate(size)
            self.assertEqual(small.vocab, direct.vocab)
            self.assertEqual(small.merges, direct.merges)
            self.assertEqual(small.encode(text[:2000]), direct.encode(text[:2000]))
        self.assertEqual(big.truncate(big.vocab_size).merges, big.merges)
        with self.assertRaises(ValueError):
            big.truncate(256)
        with self.assertRaises(ValueError):
            big.truncate(big.vocab_size + 1)

    def test_load_warns_on_unicode_version_mismatch(self):
        tok = self.train("cat cat car")
        path = Path(self.temp.name) / "tokenizer.json"
        tok.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["unicode_version"] = "0.0.0"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertWarns(RuntimeWarning):
            restored = BPETokenizer.load(path)
        self.assertEqual(restored.merges, tok.merges)

    def test_matches_slow_reference(self):
        rng = random.Random(336)
        for _ in range(100):
            words = [
                "".join(rng.choice("aaabbcdef") for _ in range(rng.randint(1, 12)))
                for _ in range(rng.randint(1, 35))
            ]
            text = " ".join(words) + "<|endoftext|>café café🙂"
            size = rng.randint(258, 290)
            tok = self.train(text, size)
            expected_vocab, expected_merges = reference_train(text, size, tok.special_tokens)
            self.assertEqual(tok.vocab, expected_vocab)
            self.assertEqual(tok.merges, expected_merges)


if __name__ == "__main__":
    unittest.main()
