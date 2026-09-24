import argparse
import json


def steps_per_epoch(meta_path: str, batch_size: int, grad_accum_steps: int) -> dict:
    """Exact number of optimizer steps to cover a prepared dataset's training
    split exactly once (drop_last=True, matching the DataLoader in
    train.py), given a batch size and gradient-accumulation factor.

    train.py's training loop otherwise cycles the data loader indefinitely
    until max_steps is reached, so max_steps must be set to exactly this
    value for a true single-epoch run.
    """
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    n_train_chunks = meta["n_train_chunks"]
    micro_batches_per_epoch = n_train_chunks // batch_size
    max_steps = micro_batches_per_epoch // grad_accum_steps
    tokens_per_step = batch_size * meta["seq_len"] * grad_accum_steps

    return {
        "n_train_chunks": n_train_chunks,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "seq_len": meta["seq_len"],
        "max_steps_for_one_epoch": max_steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "tokens_covered": max_steps * tokens_per_step,
        "train_tokens_available": meta["train_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the exact max_steps for one epoch over a prepared dataset's "
        "training split, given a batch size and grad_accum_steps."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing meta.json")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    args = parser.parse_args()

    import os

    result = steps_per_epoch(
        os.path.join(args.data_dir, "meta.json"), args.batch_size, args.grad_accum_steps
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
