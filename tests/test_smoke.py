import yaml

from scratch_llm.data.prepare_data import prepare
from scratch_llm.tokenizer.train_tokenizer import train as train_tokenizer
from scratch_llm.train import train as run_training

CORPUS_PATH = "data/raw/sample_corpus.txt"


def _write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_smoke_pipeline_runs_end_to_end(tmp_path):
    """Exercises tokenizer training -> data prep -> model build -> training
    loop -> checkpoint save/resume using a tiny model config, verifying the
    full pipeline runs without error at small scale."""
    (tmp_path / "tokenizer").mkdir()
    tokenizer_prefix = str(tmp_path / "tokenizer" / "spm")
    train_tokenizer(CORPUS_PATH, tokenizer_prefix, vocab_size=512, model_type="bpe")

    data_dir = str(tmp_path / "data")
    meta = prepare(
        input_path=CORPUS_PATH,
        tokenizer_path=tokenizer_prefix + ".model",
        output_dir=data_dir,
        seq_len=32,
        val_fraction=0.15,
    )
    assert meta["vocab_size"] == 512
    assert meta["n_train_chunks"] > 0
    assert meta["n_val_chunks"] > 0

    model_cfg = _load_yaml("configs/model_smoke.yaml")
    model_cfg["vocab_size"] = meta["vocab_size"]
    model_cfg_path = tmp_path / "model_smoke.yaml"
    _write_yaml(model_cfg_path, model_cfg)

    train_cfg = _load_yaml("configs/train_smoke.yaml")
    train_cfg["max_steps"] = 10
    train_cfg["out_dir"] = str(tmp_path / "checkpoints" / "smoke")
    train_cfg_path = tmp_path / "train_smoke.yaml"
    _write_yaml(train_cfg_path, train_cfg)

    result = run_training(str(model_cfg_path), str(train_cfg_path), data_dir)

    assert result["final_step"] == 10
    assert result["final_tokens_seen"] > 0
    assert result["param_breakdown"]["total"] > 0
    assert result["final_val_perplexity"] > 0

    ckpt_path = tmp_path / "checkpoints" / "smoke" / "ckpt_latest.pt"
    assert ckpt_path.exists()

    # Resume from the checkpoint and confirm training continues past it
    # rather than restarting, and that tokens_seen keeps accumulating.
    train_cfg["max_steps"] = 20
    train_cfg_path2 = tmp_path / "train_smoke_resume.yaml"
    _write_yaml(train_cfg_path2, train_cfg)

    result2 = run_training(
        str(model_cfg_path), str(train_cfg_path2), data_dir, resume=str(ckpt_path)
    )
    assert result2["final_step"] == 20
    assert result2["final_tokens_seen"] > result["final_tokens_seen"]
