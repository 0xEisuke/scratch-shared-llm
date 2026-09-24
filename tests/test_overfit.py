import json

import numpy as np
import yaml

from scratch_llm.data.prepare_data import prepare
from scratch_llm.tokenizer.train_tokenizer import train as train_tokenizer
from scratch_llm.train import train as run_training

N_OVERFIT_CHUNKS = 4

CORPUS_PATH = "data/raw/sample_corpus.txt"


def _write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_model_can_overfit_small_dataset(tmp_path):
    """Sanity check: a tiny model trained on a handful of fixed chunks should
    be able to drive training loss/perplexity down close to the theoretical
    minimum, confirming the model + training loop are wired up correctly."""
    (tmp_path / "tokenizer").mkdir()
    tokenizer_prefix = str(tmp_path / "tokenizer" / "spm")
    train_tokenizer(CORPUS_PATH, tokenizer_prefix, vocab_size=512, model_type="bpe")

    data_dir = str(tmp_path / "data")
    meta = prepare(
        input_path=CORPUS_PATH,
        tokenizer_path=tokenizer_prefix + ".model",
        output_dir=data_dir,
        seq_len=32,
        val_fraction=0.0,
    )
    assert meta["n_val_chunks"] == 0

    # Keep only a handful of chunks so a tiny model can clearly memorize
    # them within a small step budget, then point "val" at the same data
    # the model trains on: overfitting is measured as loss on the training
    # set itself, not a held-out split.
    dtype = np.dtype(meta["dtype"])
    chunk_len = meta["chunk_len"]
    arr = np.fromfile(f"{data_dir}/train.bin", dtype=dtype).reshape(-1, chunk_len)
    n_keep = min(N_OVERFIT_CHUNKS, arr.shape[0])
    subset = arr[:n_keep]
    subset.tofile(f"{data_dir}/train.bin")
    subset.tofile(f"{data_dir}/val.bin")
    meta["n_train_chunks"] = int(n_keep)
    meta["train_tokens"] = int(subset.size)
    with open(f"{data_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    model_cfg = _load_yaml("configs/model_smoke.yaml")
    model_cfg["vocab_size"] = meta["vocab_size"]
    model_cfg_path = tmp_path / "model_smoke.yaml"
    _write_yaml(model_cfg_path, model_cfg)

    train_cfg = _load_yaml("configs/train_overfit.yaml")
    train_cfg["out_dir"] = str(tmp_path / "checkpoints" / "overfit")
    train_cfg_path = tmp_path / "train_overfit.yaml"
    _write_yaml(train_cfg_path, train_cfg)

    result = run_training(str(model_cfg_path), str(train_cfg_path), data_dir)

    assert result["final_val_loss"] < 1.0
    assert result["final_val_perplexity"] < 3.0
