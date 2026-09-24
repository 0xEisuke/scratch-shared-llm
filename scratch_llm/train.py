import argparse
import json
import math
import os
import time
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader

from scratch_llm.checkpoint import load_checkpoint, save_checkpoint
from scratch_llm.config import ModelConfig, TrainConfig, load_yaml_config
from scratch_llm.data.dataset import ChunkedTokenDataset
from scratch_llm.evaluate import evaluate
from scratch_llm.model.params import count_parameters, format_breakdown
from scratch_llm.model.transformer import Transformer

_DTYPES = {"bf16": torch.bfloat16, "fp32": None}


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def cycle(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    epoch = 0
    while True:
        epoch += 1
        if epoch > 1:
            print(f"warning: starting epoch {epoch} over this data loader (dataset is repeating)")
        for batch in loader:
            yield batch


def build_model_and_optimizer(model_cfg: ModelConfig, train_cfg: TrainConfig, device: torch.device):
    model = Transformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    return model, optimizer


def train(
    model_config_path: str,
    train_config_path: str,
    data_dir: str,
    resume: str = None,
    out_dir: str = None,
    run_name: str = None,
    results_path: str = None,
    max_steps: int = None,
    batch_size: int = None,
    grad_accum_steps: int = None,
) -> dict:
    model_cfg = load_yaml_config(model_config_path, ModelConfig)
    train_cfg = load_yaml_config(train_config_path, TrainConfig)

    # CLI overrides let one shared train-config file (e.g. train_compare.yaml)
    # drive several runs that each need their own out_dir/run_name, without
    # hand-editing/duplicating the yaml per run.
    if out_dir is not None:
        train_cfg.out_dir = out_dir
    if run_name is not None:
        train_cfg.run_name = run_name
    if results_path is not None:
        train_cfg.results_path = results_path
    if max_steps is not None:
        train_cfg.max_steps = max_steps
    if batch_size is not None:
        train_cfg.batch_size = batch_size
    if grad_accum_steps is not None:
        train_cfg.grad_accum_steps = grad_accum_steps

    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    amp_dtype = _DTYPES[train_cfg.dtype] if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, optimizer = build_model_and_optimizer(model_cfg, train_cfg, device)
    breakdown = count_parameters(model)
    print(format_breakdown(breakdown))

    train_ds = ChunkedTokenDataset(data_dir, "train")
    val_ds = ChunkedTokenDataset(data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, drop_last=False)
    train_iter = cycle(train_loader)

    step = 0
    tokens_seen = 0
    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, model, optimizer, map_location=str(device))
        step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        print(f"Resumed from {resume} at step={step} tokens_seen={tokens_seen}")

    os.makedirs(train_cfg.out_dir, exist_ok=True)
    tokens_per_optim_step = train_cfg.batch_size * train_ds.seq_len * train_cfg.grad_accum_steps

    model.train()
    t0 = time.time()
    history = []

    while step < train_cfg.max_steps:
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(x, y)
            scaled_loss = loss / train_cfg.grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item() / train_cfg.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_optim_step

        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps:
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                f"grad_norm {grad_norm:.3f} | tokens_seen {tokens_seen:,} | {elapsed:.1f}s",
                flush=True,
            )
            history.append({"step": step, "train_loss": accum_loss, "tokens_seen": tokens_seen})

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            val_metrics = evaluate(model, val_loader, device, amp_dtype, max_batches=train_cfg.eval_iters)
            print(
                f"  eval @ step {step}: loss {val_metrics['loss']:.4f} "
                f"ppl {val_metrics['perplexity']:.3f} ({val_metrics['tokens']} tokens)",
                flush=True,
            )
            if history:
                history[-1]["val_loss"] = val_metrics["loss"]
                history[-1]["val_perplexity"] = val_metrics["perplexity"]

        if step % train_cfg.save_interval == 0 or step == train_cfg.max_steps:
            ckpt_path = os.path.join(train_cfg.out_dir, f"ckpt_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, model_cfg, train_cfg)
            latest_path = os.path.join(train_cfg.out_dir, "ckpt_latest.pt")
            save_checkpoint(latest_path, model, optimizer, step, tokens_seen, model_cfg, train_cfg)

    final_val = evaluate(model, val_loader, device, amp_dtype)
    train_time_sec = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    )

    result = {
        "final_step": step,
        "final_tokens_seen": tokens_seen,
        "final_val_loss": final_val["loss"],
        "final_val_perplexity": final_val["perplexity"],
        "param_breakdown": breakdown,
        "history": history,
        "train_time_sec": train_time_sec,
        "peak_vram_mb": peak_vram_mb,
    }

    if train_cfg.results_path and train_cfg.run_name:
        os.makedirs(os.path.dirname(train_cfg.results_path) or ".", exist_ok=True)
        row = {
            "run_name": train_cfg.run_name,
            "d_model": model_cfg.d_model,
            "attn_sharing": model_cfg.attn_sharing,
            "ffn_sharing": model_cfg.ffn_sharing,
            "total_params": breakdown["total"],
            "val_loss": final_val["loss"],
            "val_perplexity": final_val["perplexity"],
            "train_time_sec": train_time_sec,
            "peak_vram_mb": peak_vram_mb,
            "tokens_seen": tokens_seen,
        }
        with open(train_cfg.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a from-scratch decoder-only Transformer")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out-dir", default=None, help="Override train-config out_dir")
    parser.add_argument("--run-name", default=None, help="Override train-config run_name")
    parser.add_argument("--results-path", default=None, help="Override train-config results_path")
    parser.add_argument("--max-steps", type=int, default=None, help="Override train-config max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train-config batch_size")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None, help="Override train-config grad_accum_steps"
    )
    args = parser.parse_args()

    train(
        args.model_config,
        args.train_config,
        args.data_dir,
        args.resume,
        out_dir=args.out_dir,
        run_name=args.run_name,
        results_path=args.results_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )


if __name__ == "__main__":
    main()
