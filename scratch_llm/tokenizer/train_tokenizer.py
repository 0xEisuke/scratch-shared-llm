import argparse

import sentencepiece as spm


def train(
    input_path: str,
    model_prefix: str,
    vocab_size: int = 16000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> None:
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer")
    parser.add_argument("--input", required=True, help="Path to raw text file (one doc/line)")
    parser.add_argument("--model-prefix", required=True, help="Output prefix, e.g. tokenizer/spm")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--character-coverage", type=float, default=1.0)
    args = parser.parse_args()

    train(
        input_path=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )
    print(f"Trained tokenizer -> {args.model_prefix}.model / {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
