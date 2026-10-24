BEVOTRON BYTE-LEVEL BPE TOKENIZER

Python 3.10 or newer. No third-party dependencies are required for these files.
Existing regex/pytest installations are fine; tests also run with unittest.

COPY INTO YOUR EXISTING BEVOTRON REPO
  tokenizer.py                  (new file in repo root)
  train_tokenizer.py            (replace your earlier practice script)
  truncate_tokenizer.py         (new file: derive smaller vocabularies)
  tests/test_tokenizer.py        (new test file)

RUN FROM THE BEVOTRON ROOT IN POWERSHELL
  .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_tokenizer.py -v
  .\.venv\Scripts\python.exe train_tokenizer.py --input data/tokenizer_debug.txt --vocab-size 270

The tiny 'cat cat car' sample will stop below the vocabulary limit because it
has very few distinct byte pairs. This is expected, not a training failure.
The output is artifacts/tokenizer/tokenizer.json. It stores the vocabulary,
ordered merges, special tokens, and format/Unicode metadata in one JSON file.

TRAIN ON TINYSTORIES
Use a local UTF-8 TinyStories TRAINING text file with <|endoftext|> document
separators, such as TinyStoriesV2-GPT4-train.txt from the assignment README.
Do not train on the held-out validation data used to evaluate the final model.

Start with a prefix to debug before a full corpus run:
  .\.venv\Scripts\python.exe train_tokenizer.py --input data/tinystories/TinyStoriesV2-GPT4-train.txt --vocab-size 1000 --max-segments 1000

Then train the tokenizer on the complete training text:
  .\.venv\Scripts\python.exe train_tokenizer.py --input data/tinystories/TinyStoriesV2-GPT4-train.txt --vocab-size 10000

Training only learns the tokenizer. No GPU or neural-network training is used.
The implementation uses incremental pair counts, but runtime and peak RAM on
the full TinyStories corpus have not been benchmarked. Unique pre-token counts
and indexes remain in RAM. Files without special-token separators may require
buffering the entire input. Profile a subset before a complete run.

DERIVE SMALLER VOCABULARIES (VOCAB STUDY)
Train once at the largest size, then truncate; every size shares the same
merges and token IDs as a tokenizer trained directly at that size:
  .\.venv\Scripts\python.exe train_tokenizer.py --input data/tinystories/TinyStoriesV2-GPT4-train.txt --vocab-size 50000 --output artifacts/tokenizer/tokenizer_50k.json
  .\.venv\Scripts\python.exe truncate_tokenizer.py --input artifacts/tokenizer/tokenizer_50k.json --sizes 25000 16000 12000 8000 4000
Or from Python: tok.truncate(8000).

Loading a tokenizer with a different Unicode database version warns instead
of failing. Character classification and token IDs may differ across versions.
Use matching Python/Unicode versions for reproducible experiments.

USE THE TOKENIZER FROM PYTHON
  from tokenizer import BPETokenizer
  tok = BPETokenizer.load('artifacts/tokenizer/tokenizer.json')
  ids = tok.encode('Hello, world!')
  print(ids)
  print(tok.decode(ids))
  print(tok.vocab_size, tok.eos_token_id)

API
  train_bpe(input_path, vocab_size, special_tokens)
      -> (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])
  BPETokenizer(vocab, merges, special_tokens)
  tok.encode(text) -> list[int]
  tok.decode(ids, errors='replace') -> str
  tok.encode_iterable(strings) -> iterator of IDs
  tok.save(path)
  BPETokenizer.load(path)

encode_iterable encodes each supplied string independently. Supply complete
documents/records, not arbitrary character chunks if identical whole-document
token IDs are required. All valid Unicode strings round-trip through encode
and decode. Invalid UTF-8 generated token sequences use replacement characters
by default; decode(..., errors='strict') instead raises an error.

SPECIAL TOKENS AND BOUNDARIES
The default special token is <|endoftext|>, assigned ID 256. Special tokens
are never split or included in training pair counts. Longest special-token
matches take precedence. No end token is automatically appended by encode:
your dataset-preparation script must add it once per document if absent.
Special tokens must be distinct from the 256 base-byte entries.

Pre-tokenization follows the GPT-2 Unicode pattern using Python's Unicode
categories directly rather than the regex package. Loading checks the Unicode
database version so classification does not silently change across Python
versions. Merges respect pre-token and document boundaries. Frequency ties
select the lexicographically greatest byte pair, matching the handout rule.

LATER: CONNECT TO KEVIN'S FRAMEWORK
1. Update data/tinystories/prepare.py to load the saved tokenizer rather than
   the pretrained GPT-Neo tokenizer.
2. Encode train/validation separately and regenerate both binary datasets.
3. Record vocab size, EOS ID, and tokenizer identity in dataset metadata.
4. Pass the EOS/BOS IDs into model_neo.py instead of hardcoding 50256.
5. Check every ID fits the selected binary dtype. uint16 supports IDs 0..65535.

