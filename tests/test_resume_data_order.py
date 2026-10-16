import torch
import yaml

from scratch_llm.data.prepare_data import prepare
from scratch_llm.data.sampler import ResumableShuffleBatchSampler
from scratch_llm.tokenizer.train_tokenizer import train as train_tokenizer
from scratch_llm.train import train as run_training

CORPUS_PATH = "data/raw/sample_corpus.txt"


def _write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _take(sampler, n):
    it = iter(sampler)
    return [next(it) for _ in range(n)]


def test_sampler_resumes_at_the_same_position_as_an_uninterrupted_run():
    n_samples, batch_size, seed = 103, 8, 1337
    batches_per_epoch = n_samples // batch_size
    total = batches_per_epoch * 3 + 5  # spans several epoch boundaries
    full = _take(ResumableShuffleBatchSampler(n_samples, batch_size, seed), total)

    for start in (1, batches_per_epoch - 1, batches_per_epoch, batches_per_epoch + 3, total - 1):
        # Global RNG state must not matter: a resumed process has consumed a
        # different amount of it than the original one had at this point.
        torch.manual_seed(start)
        resumed = _take(
            ResumableShuffleBatchSampler(n_samples, batch_size, seed, start_batch=start),
            total - start,
        )
        assert resumed == full[start:]

    # Every epoch covers distinct samples (drop_last) and epochs differ.
    epochs = [full[e * batches_per_epoch : (e + 1) * batches_per_epoch] for e in range(3)]
    for epoch in epochs:
        flat = [i for batch in epoch for i in batch]
        assert len(set(flat)) == batches_per_epoch * batch_size
    assert epochs[0] != epochs[1]

    assert _take(ResumableShuffleBatchSampler(n_samples, batch_size, seed + 1), 1) != full[:1]


def test_interrupted_and_resumed_training_matches_uninterrupted(tmp_path):
    """Resuming from a mid-run checkpoint must land on the same weights as a
    run that was never interrupted, i.e. it sees the same batches in the
    same order rather than replaying the shuffle from the start."""
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

    model_cfg = _load_yaml("configs/model_smoke.yaml")
    model_cfg["vocab_size"] = meta["vocab_size"]
    model_cfg_path = str(tmp_path / "model_smoke.yaml")
    _write_yaml(model_cfg_path, model_cfg)

    train_cfg = _load_yaml("configs/train_smoke.yaml")
    # CPU/fp32 so the two runs are comparable bit-for-bit.
    train_cfg.update(device="cpu", dtype="fp32", max_steps=30, save_interval=6, log_interval=1)
    # Several epochs' worth of batches, so the resume point is past a wrap.
    assert 30 * train_cfg["grad_accum_steps"] > 2 * (meta["n_train_chunks"] // train_cfg["batch_size"])
    train_cfg_path = str(tmp_path / "train.yaml")
    _write_yaml(train_cfg_path, train_cfg)

    full_dir = tmp_path / "full"
    full = run_training(model_cfg_path, train_cfg_path, data_dir, out_dir=str(full_dir))

    for resume_step in (6, 18):
        resumed_dir = tmp_path / f"resumed{resume_step}"
        resumed = run_training(
            model_cfg_path,
            train_cfg_path,
            data_dir,
            resume=str(full_dir / f"ckpt_step{resume_step}.pt"),
            out_dir=str(resumed_dir),
        )

        assert resumed["final_step"] == full["final_step"]
        assert resumed["final_tokens_seen"] == full["final_tokens_seen"]
        full_losses = {h["step"]: h["train_loss"] for h in full["history"]}
        for h in resumed["history"]:
            assert abs(h["train_loss"] - full_losses[h["step"]]) < 1e-5

        a = torch.load(full_dir / "ckpt_step30.pt", weights_only=False)["model"]
        b = torch.load(resumed_dir / "ckpt_step30.pt", weights_only=False)["model"]
        for key in a:
            assert torch.allclose(a[key], b[key], atol=1e-6), key
