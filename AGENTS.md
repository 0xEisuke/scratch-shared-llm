# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

From-scratch pretraining of a Llama-style decoder-only Transformer (~0.5B params): RMSNorm, RoPE,
causal self-attention, SwiGLU FFN, all weights randomly initialized (no pretrained checkpoints).
Full spec in `mdfiles/prd.md` (Japanese).

All three sharing variants from the PRD are implemented via `ModelConfig.attn_sharing`/`ffn_sharing`
(`"none"` or `"full"`; see "Weight-sharing design" below):

1. Standard model — all layers independent
2. Attention AND FFN shared across all layers (`attn_sharing: full, ffn_sharing: full`)
3. Only attention shared across all layers; FFN and LayerNorm stay independent per layer
   (`attn_sharing: full, ffn_sharing: none`)

The 0.5B single-model pipeline (below) intentionally does **not** run full-scale training — only
pipeline correctness at small scale (smoke test + overfit check). Full-scale target is ~20x tokens
per param (~10B tokens for the 0.5B model); `configs/train_0.5b.yaml` has placeholder `max_steps`
that must be recalculated once a real corpus/batch/seq_len are chosen.

### Standard-vs-shared-weight comparison experiment

A second track (`configs/model_256/512/768/1024.yaml` + `configs/model_1024_shared_full/attn.yaml`)
compares 4 standard model sizes against the 2 sharing variants at d_model=1024, all trained on an
identical 1B-token Japanese corpus for exactly 1 epoch, seq_len=512. See `mdfiles/prd.md` for the
full spec. Key pieces specific to this track:

- **Tokenizer**: GPT-NeoX-Japanese (`abeja/gpt-neox-japanese-2.7b`, vocab=32,000, bos=31996,
  eos=pad=unk=31999 `<|endoftext|>`) — chosen because its vocab size matches the PRD's approximate
  parameter-count table almost exactly (LLM-jp v2.2 does not). Fetch it once with
  `python -m scratch_llm.tokenizer.fetch_pretrained_tokenizer --output-dir tokenizers/gpt-neox-japanese`,
  then point `--tokenizer` at that directory. `scratch_llm/tokenizer/tokenizer.py`'s `Tokenizer` class
  dispatches on the path: a `.model` file uses the original SentencePiece backend (self-trained
  tokenizers, existing tests); anything else (a local dir or hub repo id) loads via
  `transformers.AutoTokenizer`.
- **Corpus**: none is checked into the repo (`data/raw/sample_corpus.txt` is a 52-line English toy
  corpus used only by `tests/`). `scratch_llm/data/download_corpus.py` streams a public dataset
  (default: `wikimedia/wikipedia` 20231101.ja) down to a raw text file without pulling the whole
  dataset. `prepare_data.py --max-tokens N` then truncates the *tokenized* stream to exactly N tokens,
  so every model in the comparison can point at the same prepared data-dir and see an identical
  token stream.
- **Packed-sequence mask**: `ModelConfig.eos_token_id`, when set, makes `Transformer.forward` build a
  per-batch boolean attention mask (`model/packed_mask.py::build_packed_attn_mask`) from the input's
  EOS positions and pass it through every block/attention call, blocking attention across document
  boundaries within a packed 513-token chunk in addition to the ordinary causal constraint. Leaving
  it `None` (the default, used by `configs/model_0.5b.yaml`/`configs/model_smoke.yaml`) keeps the
  plain `is_causal=True` SDPA path.
- **1-epoch step count**: `train.py`'s loop cycles its `DataLoader` indefinitely up to `max_steps`
  (printing a warning if it wraps around), so `max_steps` must be computed to land exactly one epoch:
  `python -m scratch_llm.compute_epoch_steps --data-dir <dir> --batch-size N --grad-accum-steps M`.
- **Per-run metrics + comparison plot**: `train.py` accepts `--out-dir`/`--run-name`/`--results-path`/
  `--max-steps` overrides so one shared `configs/train_compare.yaml` drives all 6 runs. When
  `run_name`+`results_path` are set, a JSONL row (params, val loss/perplexity, wall-clock train time,
  peak VRAM, tokens seen) is appended at the end of training. `scratch_llm/plot_results.py` reads that
  JSONL and plots total_params vs val_loss for the 4 standard points plus the 2 shared-variant points.

## Environment

- Python venv at `.venv/`, activate or call `.venv/Scripts/python` (Windows) directly.
- `pip install -r requirements.txt` installs torch, sentencepiece, numpy, pyyaml, tqdm, pytest.
- Plain `pip install torch` on this machine resolves to a CPU-only wheel. For CUDA (RTX 4090
  available), install explicitly:
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- No `pyproject.toml`/package install step needed — always invoke modules with `python -m` from the
  repo root (not `python path/to/file.py`), so `scratch_llm` resolves as a package via cwd on
  `sys.path`.

## Commands

Run all from the repo root.

```
# Tests (smoke pipeline + overfit sanity check)
python -m pytest tests/ -v

# Train a tokenizer (SentencePiece BPE)
python -m scratch_llm.tokenizer.train_tokenizer --input <text file> --model-prefix <out prefix> --vocab-size 16000

# Tokenize + fixed-length-chunk a corpus into memmap-friendly .bin shards
python -m scratch_llm.data.prepare_data --input <text file> --tokenizer <prefix>.model --output-dir <dir> --seq-len <N>

# Print exact parameter count + breakdown for a model config
python -m scratch_llm.report_model_size --model-config configs/model_0.5b.yaml

# Train (also used for smoke/overfit runs with configs/*_smoke.yaml / *_overfit.yaml)
python -m scratch_llm.train --model-config configs/model_0.5b.yaml --train-config configs/train_0.5b.yaml --data-dir <prepared data dir> [--resume <ckpt path>]

# --- Standard-vs-shared-weight comparison track ---

# Fetch the pretrained GPT-NeoX-Japanese tokenizer once (needs network access)
python -m scratch_llm.tokenizer.fetch_pretrained_tokenizer --output-dir tokenizers/gpt-neox-japanese

# Stream a public Japanese corpus down to raw text (see AGENTS.md notes on scale/time before running at full size)
python -m scratch_llm.data.download_corpus --output <raw txt path> --target-tokens 1000000000

# Tokenize once with --max-tokens so every model config shares an identical token stream
python -m scratch_llm.data.prepare_data --input <raw txt path> --tokenizer tokenizers/gpt-neox-japanese --output-dir <prepared dir> --seq-len 512 --max-tokens 1000000000

# Compute the exact max_steps for one epoch over that prepared dir
python -m scratch_llm.compute_epoch_steps --data-dir <prepared dir> --batch-size 64 --grad-accum-steps 1

# Train one of the 6 comparison models, overriding out_dir/run_name/max_steps per run
python -m scratch_llm.train --model-config configs/model_1024.yaml --train-config configs/train_compare.yaml --data-dir <prepared dir> --out-dir checkpoints/compare/model_1024 --run-name standard_1024 --max-steps <from compute_epoch_steps>

# After all 6 runs have appended to configs/train_compare.yaml's results_path
python -m scratch_llm.plot_results --results-path checkpoints/compare/results.jsonl --output checkpoints/compare/params_vs_val_loss.png
```

To run a single test: `python -m pytest tests/test_overfit.py -v -s` (`-s` shows the training log).

## Architecture

```
scratch_llm/
  config.py                   # ModelConfig / TrainConfig dataclasses, YAML loader
  model/
    norm.py                    # RMSNorm
    rope.py                    # RoPE tables + apply_rotary_pos_emb
    attention.py                # CausalSelfAttention (torch SDPA, is_causal=True)
    ffn.py                       # SwiGLUFFN
    block.py                      # TransformerBlock — takes attn/ffn as constructor args, threads attn_mask
    layer_assignment.py            # build_layers(config): the ONLY place sharing is decided ("none"/"full")
    packed_mask.py                  # build_packed_attn_mask: causal + EOS-boundary mask for packed sequences
    transformer.py                   # Transformer: embedding, RoPE, layers, norm, lm_head, builds packed mask
    params.py                         # count_parameters (dedups shared params by identity)
  tokenizer/
    train_tokenizer.py                # SentencePiece BPE trainer (CLI, self-trained tokenizers)
    fetch_pretrained_tokenizer.py       # CLI: download+cache a pretrained HF tokenizer, report special tokens
    tokenizer.py                         # Tokenizer: dispatches to SentencePiece (.model file) or HF AutoTokenizer
  data/
    prepare_data.py                     # tokenize -> [truncate to --max-tokens] -> chunk_tokens() -> .bin/meta.json
    download_corpus.py                   # streams a public HF dataset (default ja Wikipedia) to a raw text file
    dataset.py                            # ChunkedTokenDataset, memmap-backed
  checkpoint.py                          # save/load_checkpoint (model+optim+step+tokens_seen+configs)
  evaluate.py                            # token-weighted CE loss + perplexity over a DataLoader
  train.py                                # training loop + get_lr (warmup/cosine) + main(); VRAM/time/JSONL results logging
  compute_epoch_steps.py                  # CLI: exact max_steps for one epoch over a prepared data-dir
  plot_results.py                          # CLI: total_params vs val_loss plot from a results.jsonl
  report_model_size.py                      # CLI: build a config, print param breakdown
configs/                                    # model_0.5b / model_smoke, train_0.5b / train_smoke / train_overfit,
                                             # model_256/512/768/1024 + model_1024_shared_full/attn, train_compare
data/raw/sample_corpus.txt                    # small fixed ENGLISH corpus used only by tests/ (not the ja comparison)
tests/test_smoke.py                            # tokenizer -> data prep -> train -> checkpoint/resume, tiny scale
tests/test_overfit.py                           # trains on a handful of chunks, asserts loss -> ~0 (ppl -> 1)
tests/test_packed_mask.py                        # build_packed_attn_mask correctness (EOS boundaries, batching)
tests/test_sharing_variants.py                    # forward/backward + checkpoint round-trip for all 3 variants
tests/test_data_pipeline.py                        # chunk shift semantics, --max-tokens truncation, dataset getitem
```

### Weight-sharing design

`TransformerBlock` (`model/block.py`) receives its `attn` and `ffn` submodules as constructor
arguments rather than constructing them — this is what lets one module instance be reused across
multiple blocks. `model/layer_assignment.py::build_layers` is the single place that decides, per
layer index, which attention/FFN instance to use, keyed by `ModelConfig.attn_sharing`/`ffn_sharing`:
`"none"` gives every layer its own instance, `"full"` reuses one instance across all `n_layers`
(`layer_assignment.py::_sharing_key`). `attn_norm`/`ffn_norm` are always constructed fresh per
`TransformerBlock`, so LayerNorm stays independent per layer regardless of sharing mode (needed for
variant 3). To add a grouped/block-sharing mode beyond `"none"`/`"full"`, extend `_sharing_key` to
return a group id per layer index — no change to `Attention`, `FFN`, `TransformerBlock`, or
`Transformer` is needed. `model/params.py::count_parameters` and PyTorch's own `state_dict()`/
`named_modules()` already de-duplicate by module identity, so total/breakdown param counts and
checkpoint save/load stay correct once modules are shared (verified in `tests/test_sharing_variants.py`).

### Data pipeline

`prepare_data.py` tokenizes a corpus line-by-line (each line gets an EOS appended), concatenates ids,
then `chunk_tokens()` splits the stream into **non-overlapping** fixed-length chunks of `seq_len + 1`
tokens (remainder dropped) — one chunk yields both the input (`chunk[:-1]`) and shifted target
(`chunk[1:]`). Chunks are written as raw `uint16`/`uint32` (dtype picked from vocab size) via
`np.ndarray.tofile`, and `ChunkedTokenDataset` reads them back with `np.memmap` — no full corpus is
loaded into RAM.

### Training loop notes

- `TrainConfig.dtype` ("bf16"/"fp32") drives `torch.autocast`; on CPU (no CUDA) autocast is disabled
  regardless of the config, since these configs are written for the CUDA case.
- Gradient accumulation happens inside one optimizer step (loop over `grad_accum_steps` micro-batches,
  `loss / grad_accum_steps` before `.backward()`), then a single `clip_grad_norm_` + `optimizer.step()`.
- `tokens_seen` is `batch_size * dataset.seq_len * grad_accum_steps` per optimizer step, persisted in
  checkpoints and resumed from there — not reset to 0 on `--resume`.
- `evaluate()` (`evaluate.py`) computes loss weighted by token count (sum of `loss * n_tokens` divided
  by total tokens, then `exp`), not a mean-of-batch-means, so it stays correct if batch token counts
  ever differ.
- Checkpoints save both `model_config`/`train_config` dicts alongside `model`/`optimizer` state, plus
  `step` and `tokens_seen`, under `TrainConfig.out_dir` as `ckpt_step<N>.pt` and `ckpt_latest.pt`.

### 0.5B config (`configs/model_0.5b.yaml`)

d_model=1536, n_layers=16, n_heads=24 (head_dim=64), d_ff=4096 (SwiGLU), vocab_size=16000,
max_seq_len=2048, untied embeddings. Exact parameter count (via
`python -m scratch_llm.report_model_size --model-config configs/model_0.5b.yaml`): **502,187,520**
(tok_embeddings 24,576,000 / attention 150,994,944 / ffn 301,989,888 / norms 50,688 / lm_head
24,576,000).

`configs/model_smoke.yaml` is a tiny stand-in (d_model=64, n_layers=2) used only by the tests; its
`vocab_size` must match whatever tokenizer the test trains (tests train one with `vocab_size=512` and
override the loaded config's `vocab_size` before writing it to a temp file — see `tests/test_smoke.py`).

### Comparison-experiment configs (`configs/model_{256,512,768,1024}.yaml` + `*_shared_{full,attn}.yaml`)

n_layers=16, n_heads=8 (fixed), d_ff = `round((8/3 * d_model) / 256) * 256` (rounding rule fixed across
all 4 sizes), vocab_size=32,000, max_seq_len=512, tied embeddings, eos_token_id=31999. Exact parameter
counts (via `report_model_size.py`, weight-tied):

| d_model | d_ff  | attn_sharing | ffn_sharing | total params  |
| ------: | ----: | :----------- | :---------- | -------------:|
|     256 |   768 | none         | none        |     21,831,936 |
|     512 |  1280 | none         | none        |     64,635,392 |
|     768 |  2048 | none         | none        |    137,847,552 |
|   1,024 |  2816 | none         | none        |    238,322,688 |
|   1,024 |  2816 | full         | full        |     45,646,848 |
|   1,024 |  2816 | full         | none        |    175,408,128 |
