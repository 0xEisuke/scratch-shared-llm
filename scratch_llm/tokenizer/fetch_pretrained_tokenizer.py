import argparse
import json
import os


def fetch(repo_id: str, output_dir: str) -> dict:
    """Download a pretrained Hugging Face tokenizer and cache it locally.

    Also verifies the special tokens (bos/eos/pad) and reports the real
    vocab size, since `ModelConfig.vocab_size` / `eos_token_id` must match
    the tokenizer exactly (embedding table size, packed-sequence boundary
    detection).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    os.makedirs(output_dir, exist_ok=True)
    tok.save_pretrained(output_dir)

    report = {
        "repo_id": repo_id,
        "output_dir": output_dir,
        "vocab_size": len(tok),
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "unk_token": tok.unk_token,
        "unk_token_id": tok.unk_token_id,
    }
    with open(os.path.join(output_dir, "fetch_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and cache a pretrained Hugging Face tokenizer for local, "
        "offline use by scratch_llm.tokenizer.tokenizer.Tokenizer"
    )
    parser.add_argument(
        "--repo-id",
        default="abeja/gpt-neox-japanese-2.7b",
        help="Hugging Face hub repo id to pull the tokenizer files from",
    )
    parser.add_argument("--output-dir", required=True, help="Local directory to save the tokenizer into")
    args = parser.parse_args()

    report = fetch(args.repo_id, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
