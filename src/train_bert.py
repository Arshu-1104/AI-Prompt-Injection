"""Fine-tune DistilBERT as the 'bert' model.

    python -m src.train_bert --epochs 3 --batch-size 16 --seed 42
    python -m src.train_bert --max-samples 64 --epochs 1     # CPU smoke test
    python -m src.train_bert --push-to-hub <user>/promptguard-bert   # then set BERT_MODEL_ID
"""
from __future__ import annotations

from src.transformer_training import ModelSpec, build_arg_parser, run_cli, train_model

SPEC = ModelSpec(name="bert", base_model="distilbert-base-uncased", output_dirname="bert_model")


def main(argv: list[str] | None = None) -> dict:
    return train_model(SPEC, build_arg_parser(SPEC).parse_args(argv))


if __name__ == "__main__":
    run_cli(main)
