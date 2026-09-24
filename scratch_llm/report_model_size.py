import argparse

from scratch_llm.config import ModelConfig, load_yaml_config
from scratch_llm.model.params import count_parameters, format_breakdown
from scratch_llm.model.transformer import Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a model from a config and report its parameter count")
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.model_config, ModelConfig)
    model = Transformer(cfg)
    breakdown = count_parameters(model)

    print(f"Config: {args.model_config}")
    print(cfg)
    print()
    print(format_breakdown(breakdown))


if __name__ == "__main__":
    main()
