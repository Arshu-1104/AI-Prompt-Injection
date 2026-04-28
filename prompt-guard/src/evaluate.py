from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.preprocess import get_train_test_data

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception as exc:
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)


LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]


def _predict_bert(texts: list[str], model_dir: Path) -> list[str]:
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            pred_idx = int(torch.argmax(logits, dim=1).item())
            predictions.append(LABELS[pred_idx])
    return predictions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    figures_dir = artifacts_dir / "figures"
    models_dir = root / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _, test_df = get_train_test_data()
    x_test = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    classical_path = models_dir / "classical_model.pkl"
    bert_dir = models_dir / "bert_model"
    if not classical_path.exists():
        raise FileNotFoundError(f"Classical model not found at {classical_path}.")
    if not bert_dir.exists():
        raise FileNotFoundError(f"BERT model not found at {bert_dir}.")

    classical_model = joblib.load(classical_path)
    classical_preds = classical_model.predict(x_test).tolist()
    bert_preds = _predict_bert(x_test, bert_dir)

    cm_classical = confusion_matrix(y_true, classical_preds, labels=LABELS)
    cm_bert = confusion_matrix(y_true, bert_preds, labels=LABELS)

    classical_report = classification_report(y_true, classical_preds, output_dict=True, zero_division=0)
    bert_report = classification_report(y_true, bert_preds, output_dict=True, zero_division=0)

    comparison_df = pd.DataFrame(
        [
            {
                "model": "classical",
                "accuracy": classical_report["accuracy"],
                "precision_weighted": classical_report["weighted avg"]["precision"],
                "recall_weighted": classical_report["weighted avg"]["recall"],
                "f1_weighted": classical_report["weighted avg"]["f1-score"],
            },
            {
                "model": "bert",
                "accuracy": bert_report["accuracy"],
                "precision_weighted": bert_report["weighted avg"]["precision"],
                "recall_weighted": bert_report["weighted avg"]["recall"],
                "f1_weighted": bert_report["weighted avg"]["f1-score"],
            },
        ]
    )

    per_class_rows: list[dict[str, float | str]] = []
    for label in LABELS:
        per_class_rows.append(
            {"class": label, "model": "classical", "f1": classical_report[label]["f1-score"]}
        )
        per_class_rows.append({"class": label, "model": "bert", "f1": bert_report[label]["f1-score"]})
    per_class_df = pd.DataFrame(per_class_rows)

    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_classical, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS
        )
        plt.title("Confusion Matrix - Classical")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(figures_dir / "confusion_matrix_classical.png", dpi=140)
        plt.close()

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_bert, annot=True, fmt="d", cmap="Purples", xticklabels=LABELS, yticklabels=LABELS
        )
        plt.title("Confusion Matrix - BERT")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(figures_dir / "confusion_matrix_bert.png", dpi=140)
        plt.close()

        melted = comparison_df.melt(id_vars="model", var_name="metric", value_name="score")
        plt.figure(figsize=(10, 5))
        sns.barplot(data=melted, x="metric", y="score", hue="model")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=20)
        plt.title("Model Comparison on Held-Out Test Set")
        plt.tight_layout()
        plt.savefig(figures_dir / "model_comparison.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=per_class_df, x="class", y="f1", hue="model")
        plt.ylim(0, 1.05)
        plt.title("Per-Class F1 Comparison")
        plt.tight_layout()
        plt.savefig(figures_dir / "per_class_f1.png", dpi=140)
        plt.close()
    else:
        print(f"Plotting unavailable due to environment policy: {PLOTTING_IMPORT_ERROR}")

    metrics_path = artifacts_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    metrics["evaluation"] = {
        "comparison": comparison_df.to_dict(orient="records"),
        "classical_per_class": classical_report,
        "bert_per_class": bert_report,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Final comparison table:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
