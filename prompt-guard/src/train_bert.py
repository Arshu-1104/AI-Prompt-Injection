from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.preprocess import get_train_test_data


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


def train_bert(random_state: int = 42) -> dict[str, float]:
    _ = random_state
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    artifacts_dir = root / "artifacts"
    bert_dir = models_dir / "bert_model"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = get_train_test_data()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["label_id"] = train_df["label"].map(LABEL_TO_ID)
    test_df["label_id"] = test_df["label"].map(LABEL_TO_ID)

    if train_df["label_id"].isna().any() or test_df["label_id"].isna().any():
        raise ValueError("Label encoding failed: unknown label found in dataset.")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
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
        output_dir=str(bert_dir / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    parsed_metrics = {
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "precision": float(metrics.get("eval_precision", 0.0)),
        "recall": float(metrics.get("eval_recall", 0.0)),
        "f1": float(metrics.get("eval_f1", 0.0)),
    }

    trainer.save_model(str(bert_dir))
    tokenizer.save_pretrained(str(bert_dir))

    metrics_path = artifacts_dir / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    metrics_data["bert_model"] = parsed_metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Saved BERT model to: {bert_dir}")
    return parsed_metrics


def main() -> None:
    train_bert()


if __name__ == "__main__":
    main()
