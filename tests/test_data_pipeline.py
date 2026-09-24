import json
import os

import numpy as np

from scratch_llm.data.dataset import ChunkedTokenDataset
from scratch_llm.data.prepare_data import chunk_tokens, tokenize_corpus, truncate_to_token_budget
from scratch_llm.tokenizer.tokenizer import Tokenizer
from scratch_llm.tokenizer.train_tokenizer import train as train_tokenizer

CORPUS_PATH = "data/raw/sample_corpus.txt"


def test_chunk_tokens_shift_semantics():
    """Each chunk is seq_len+1 tokens; input is chunk[:-1], labels is
    chunk[1:], i.e. labels[t] == input[t+1] (next-token prediction)."""
    seq_len = 4
    ids = np.arange(20, dtype=np.uint16)  # 20 = 4 chunks of length 5
    chunks = chunk_tokens(ids, seq_len)
    assert chunks.shape == (4, seq_len + 1)

    for chunk in chunks:
        x = chunk[:-1]
        y = chunk[1:]
        assert np.array_equal(y[:-1], x[1:])


def test_chunk_tokens_drops_remainder():
    seq_len = 4
    ids = np.arange(17, dtype=np.uint16)  # 17 // 5 = 3 chunks, 2 dropped
    chunks = chunk_tokens(ids, seq_len)
    assert chunks.shape == (3, seq_len + 1)


def test_truncate_to_token_budget():
    ids = np.arange(100, dtype=np.uint16)
    assert len(truncate_to_token_budget(ids, 50)) == 50
    assert len(truncate_to_token_budget(ids, 200)) == 100  # no-op if already short
    assert len(truncate_to_token_budget(ids, None)) == 100  # no-op if unset


def test_tokenize_corpus_max_tokens_early_stop_matches_full_tokenization(tmp_path):
    """tokenize_corpus streams into a growable numpy buffer rather than a
    Python list of boxed ints (which OOMs at real corpus scale -- see
    prepare_data.py). Capping with max_tokens must both early-stop (not
    tokenize the whole file) and produce a prefix identical to what full
    tokenization would give."""
    tokenizer_prefix = str(tmp_path / "spm")
    train_tokenizer(CORPUS_PATH, tokenizer_prefix, vocab_size=512, model_type="bpe")
    tok = Tokenizer(tokenizer_prefix + ".model")

    full = tokenize_corpus(CORPUS_PATH, tok)
    capped = tokenize_corpus(CORPUS_PATH, tok, max_tokens=37)

    assert len(capped) >= 37
    assert len(capped) < len(full)
    assert np.array_equal(capped, full[: len(capped)])


def test_chunked_token_dataset_getitem_shift(tmp_path):
    seq_len = 4
    chunk_len = seq_len + 1
    ids = np.arange(20, dtype=np.uint16)
    chunks = ids.reshape(-1, chunk_len)

    data_dir = tmp_path
    chunks.tofile(str(data_dir / "train.bin"))
    meta = {"seq_len": seq_len, "chunk_len": chunk_len, "dtype": "uint16"}
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    ds = ChunkedTokenDataset(str(data_dir), "train")
    assert len(ds) == 4
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
