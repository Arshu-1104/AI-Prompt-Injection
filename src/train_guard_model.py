"""Fine-tune ModernBERT-base as the 'guard' model.

    python -m src.train_guard_model --epochs 3 --batch-size 16 --seed 42
    python -m src.train_guard_model --max-samples 64 --epochs 1 --batch-size 8   # CPU smoke test
    python -m src.train_guard_model --push-to-hub <user>/promptguard-guard   # then set GUARD_MODEL_ID

Downloads answerdotai/ModernBERT-base (~600 MB) on first run; needs transformers>=4.48.
"""
from __future__ import annotations

from src.transformer_training import ModelSpec, build_arg_parser, run_cli, train_model

MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_CHOICE_REASON = (
    "ModernBERT-base is a 149M-parameter modern encoder with strong NLU results; "
    "the Hugging Face model card documents standard downstream classification "
    "fine-tuning via AutoModelForSequenceClassification."
)
SPEC = ModelSpec(
    name="guard",
    base_model=MODEL_NAME,
    output_dirname="guard_model",
    metadata={"model_choice_reason": MODEL_CHOICE_REASON},
)


def main(argv: list[str] | None = None) -> dict:
    return train_model(SPEC, build_arg_parser(SPEC).parse_args(argv))


if __name__ == "__main__":
    run_cli(main)
