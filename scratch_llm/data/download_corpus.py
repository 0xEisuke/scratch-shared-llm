import argparse

# Default estimate of Japanese characters per BPE token for a ~32k-vocab
# tokenizer (GPT-NeoX-Japanese-style). Used only to decide how much raw
# text to pull down; the exact token count actually used for training is
# enforced later and precisely by `prepare_data.py --max-tokens`, which
# tokenizes for real, so this only needs to be a safe *over*-estimate of
# how many characters are needed (i.e. err low on chars-per-token).
DEFAULT_CHARS_PER_TOKEN = 1.6


def download(
    output_path: str,
    target_tokens: int,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.ja",
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    max_articles: int = None,
    progress_every: int = 200,
) -> dict:
    """Stream a public Japanese text dataset and write it line-by-line to a
    plain text file, stopping once enough raw text has been collected to
    plausibly cover `target_tokens` after tokenization (with margin).

    Streaming (rather than a full dataset download) means only as much data
    as actually needed is pulled over the network, which matters here since
    the full dataset (e.g. all of Japanese Wikipedia) is far larger than
    the ~1B-token budget this project needs.
    """
    from datasets import load_dataset

    target_chars = int(target_tokens * chars_per_token)
    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    total_chars = 0
    n_articles = 0
    n_lines = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for example in ds:
            text = example.get("text", "")
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                out.write(line + "\n")
                total_chars += len(line)
                n_lines += 1
            n_articles += 1

            if n_articles % progress_every == 0:
                print(
                    f"articles={n_articles:,} lines={n_lines:,} chars={total_chars:,} "
                    f"(target_chars={target_chars:,})"
                )

            if total_chars >= target_chars:
                break
            if max_articles is not None and n_articles >= max_articles:
                break

    return {
        "output_path": output_path,
        "dataset": f"{dataset_name}/{dataset_config}",
        "n_articles": n_articles,
        "n_lines": n_lines,
        "total_chars": total_chars,
        "target_tokens": target_tokens,
        "target_chars": target_chars,
        "reached_target_chars": total_chars >= target_chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a public Japanese corpus (default: Japanese Wikipedia) down to "
        "a plain text file, stopping once roughly enough raw text for --target-tokens is collected. "
        "The exact token cap is enforced later by prepare_data.py --max-tokens."
    )
    parser.add_argument("--output", required=True, help="Path to write the raw text file to")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--dataset-name", default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", default="20231101.ja")
    parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional hard cap on number of articles streamed, for smoke testing",
    )
    args = parser.parse_args()

    report = download(
        output_path=args.output,
        target_tokens=args.target_tokens,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        chars_per_token=args.chars_per_token,
        max_articles=args.max_articles,
    )
    print(report)


if __name__ == "__main__":
    main()
