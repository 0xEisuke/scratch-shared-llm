import argparse
import json
import os

import numpy as np

from scratch_llm.tokenizer.tokenizer import Tokenizer


def _dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size < 2**16 else np.uint32


def tokenize_corpus(input_path: str, tokenizer: Tokenizer, max_tokens: int = None) -> np.ndarray:
    """Tokenize a corpus into a flat id array without ever materializing the
    whole stream as a Python list of boxed ints.

    A plain `ids = []; ids.extend(...)` accumulator (the original
    implementation) holds one Python `int` object per token — for a
    ~1B-token corpus that is tens of GB of pure object overhead, which
    reliably crashes the process well before `np.array(ids, ...)` is ever
    reached. Writing straight into a preallocated, doubling numpy buffer
    keeps per-token overhead down to the 2 (or 4) packed bytes of `dtype`.

    When `max_tokens` is given, tokenization stops as soon as that many
    tokens have been collected, so a corpus file much larger than the
    requested budget (kept as safety margin by the corpus downloader)
    doesn't get tokenized in full for no benefit.
    """
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    cap = max_tokens if max_tokens is not None else 1_000_000
    buf = np.empty(cap, dtype=dtype)
    pos = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_eos=True)
            n = len(ids)
            if pos + n > buf.shape[0]:
                new_cap = max(buf.shape[0] * 2, pos + n)
                grown = np.empty(new_cap, dtype=dtype)
                grown[:pos] = buf[:pos]
                buf = grown
            buf[pos : pos + n] = ids
            pos += n
            if max_tokens is not None and pos >= max_tokens:
                break

    return buf[:pos]


def truncate_to_token_budget(ids: np.ndarray, max_tokens: int) -> np.ndarray:
    """Cap a token stream to an exact length.

    Used so multiple model configs trained for comparison can all point at
    data prepared with the same `--max-tokens`, guaranteeing they see an
    identical token stream rather than merely "the same corpus file" (whose
    tokenized length can otherwise differ run to run if the corpus file
    itself is regenerated/appended to).
    """
    if max_tokens is not None and len(ids) > max_tokens:
        return ids[:max_tokens]
    return ids


def chunk_tokens(ids: np.ndarray, seq_len: int) -> np.ndarray:
    """Split a 1D token stream into non-overlapping fixed-length chunks.

    Each chunk has length `seq_len + 1` so a single chunk yields both the
    input (chunk[:-1]) and the shifted target (chunk[1:]) for causal LM
    training. Any remainder that doesn't fill a full chunk is dropped.
    """
    chunk_len = seq_len + 1
    n_chunks = len(ids) // chunk_len
    if n_chunks == 0:
        raise ValueError(
            f"Not enough tokens ({len(ids)}) to form a single chunk of length {chunk_len}"
        )
    ids = ids[: n_chunks * chunk_len]
    return ids.reshape(n_chunks, chunk_len)


def prepare(
    input_path: str,
    tokenizer_path: str,
    output_dir: str,
    seq_len: int,
    val_fraction: float = 0.1,
    max_tokens: int = None,
) -> dict:
    tokenizer = Tokenizer(tokenizer_path)
    ids = tokenize_corpus(input_path, tokenizer, max_tokens=max_tokens)
    total_tokenized = len(ids)
    ids = truncate_to_token_budget(ids, max_tokens)
    if max_tokens is not None and total_tokenized < max_tokens:
        raise ValueError(
            f"Corpus only tokenized to {total_tokenized} tokens, short of the requested "
            f"max_tokens={max_tokens}. Provide a larger input corpus."
        )

    chunks = chunk_tokens(ids, seq_len)
    if val_fraction > 0 and len(chunks) > 1:
        n_val = max(1, int(len(chunks) * val_fraction))
        n_val = min(n_val, len(chunks) - 1)
    else:
        n_val = 0

    if n_val > 0:
        train_chunks, val_chunks = chunks[:-n_val], chunks[-n_val:]
    else:
        train_chunks, val_chunks = chunks, chunks[:0]

    os.makedirs(output_dir, exist_ok=True)
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    train_chunks.astype(dtype).tofile(os.path.join(output_dir, "train.bin"))
    val_chunks.astype(dtype).tofile(os.path.join(output_dir, "val.bin"))

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_id,
        "seq_len": seq_len,
        "chunk_len": seq_len + 1,
        "dtype": np.dtype(dtype).name,
        "n_train_chunks": int(train_chunks.shape[0]),
        "n_val_chunks": int(val_chunks.shape[0]),
        "train_tokens": int(train_chunks.size),
        "val_tokens": int(val_chunks.size),
        "total_tokenized": int(total_tokenized),
        "max_tokens": max_tokens,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize + fixed-length-chunk a text corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True, help="Path to .model SentencePiece file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap the tokenized stream to exactly this many tokens before chunking "
        "(e.g. so several model configs can share one identical 1B-token corpus)",
    )
    args = parser.parse_args()

    meta = prepare(
        args.input, args.tokenizer, args.output_dir, args.seq_len, args.val_fraction, args.max_tokens
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
