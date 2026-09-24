import os
from typing import List


class Tokenizer:
    """Thin wrapper unifying two possible tokenizer backends behind one
    encode/decode/vocab_size/pad_id/bos_id/eos_id surface, so the rest of
    the pipeline (`prepare_data.py`, `dataset.py`) doesn't need to know
    which backend is in use:

    - A SentencePiece `.model` file (self-trained via `train_tokenizer.py`).
    - A pretrained Hugging Face tokenizer, given as a local directory (e.g.
      produced by `fetch_pretrained_tokenizer.py`) or a hub repo id.
    """

    def __init__(self, model_path: str):
        if model_path.endswith(".model") and os.path.isfile(model_path):
            self._backend = _SentencePieceBackend(model_path)
        else:
            self._backend = _HFBackend(model_path)

    @property
    def vocab_size(self) -> int:
        return self._backend.vocab_size

    @property
    def pad_id(self) -> int:
        return self._backend.pad_id

    @property
    def bos_id(self) -> int:
        return self._backend.bos_id

    @property
    def eos_id(self) -> int:
        return self._backend.eos_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        return self._backend.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: List[int]) -> str:
        return self._backend.decode(ids)


class _SentencePieceBackend:
    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)


class _HFBackend:
    """Wraps a `transformers` tokenizer (local dir or hub repo id)."""

    def __init__(self, path_or_repo_id: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path_or_repo_id)

    @property
    def vocab_size(self) -> int:
        # len() rather than the .vocab_size attribute: some tokenizers carry
        # extra added/special tokens past the base vocab, and this must
        # equal the highest possible token id + 1 for the embedding table.
        return len(self.tok)

    @property
    def pad_id(self):
        return self.tok.pad_token_id

    @property
    def bos_id(self):
        return self.tok.bos_token_id

    @property
    def eos_id(self):
        return self.tok.eos_token_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)
