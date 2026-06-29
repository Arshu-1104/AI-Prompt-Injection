from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import transformers

try:
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    GUARD_AVAILABLE = True
except Exception:
    GUARD_AVAILABLE = False

from src.preprocess import get_train_test_data


MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_CHOICE_REASON = (
    "ModernBERT-base is a 149M-parameter modern encoder with strong NLU results; "
    "the Hugging Face model card documents standard downstream classification "
    "fine-tuning via AutoModelForSequenceClassification."
)
LABEL_TO_ID = {"SAFE": 0, "SUSPICIOUS": 1, "MALICIOUS": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def train_guard_model(random_state: int = 42) -> dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    artifacts_dir = root / "artifacts"
    guard_dir = models_dir / "guard_model"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not GUARD_AVAILABLE:
        metrics_path = artifacts_dir / "metrics.json"
        metrics_data = {}
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        metrics_data["guard_model"] = {
            "skipped": True,
            "reason": "Torch/Transformers blocked by Windows application control policy.",
            "model_name": MODEL_NAME,
            "model_choice_reason": MODEL_CHOICE_REASON,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print("Skipping guard model training due to blocked runtime dependencies.")
        return metrics_data["guard_model"]

    _ = random_state
    train_df, test_df = get_train_test_data()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["label_id"] = train_df["label"].map(LABEL_TO_ID)
    test_df["label_id"] = test_df["label"].map(LABEL_TO_ID)

    if train_df["label_id"].isna().any() or test_df["label_id"].isna().any():
        raise ValueError("Label encoding failed: unknown label found in dataset.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    eval_dataset = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    eval_dataset = eval_dataset.map(tokenize_batch, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        output_dir=str(guard_dir / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
    )
    if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 46):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    parsed_metrics = {
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "precision": float(metrics.get("eval_precision", 0.0)),
        "recall": float(metrics.get("eval_recall", 0.0)),
        "f1": float(metrics.get("eval_f1", 0.0)),
        "model_name": MODEL_NAME,
        "model_choice_reason": MODEL_CHOICE_REASON,
    }

    trainer.save_model(str(guard_dir))
    tokenizer.save_pretrained(str(guard_dir))

    metrics_path = artifacts_dir / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    metrics_data["guard_model"] = parsed_metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Saved guard model to: {guard_dir}")
    return parsed_metrics


def main() -> None:
    train_guard_model()


if __name__ == "__main__":
    main()
