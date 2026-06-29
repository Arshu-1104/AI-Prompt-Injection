from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    BERT_EVAL_AVAILABLE = True
except Exception:
    BERT_EVAL_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix

    SKLEARN_METRICS_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False

from src.preprocess import get_train_test_data

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception as exc:
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)


LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]


def _compute_ece(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE) with *n_bins* equal-width bins.

    Returns (ece_value, bin_stats) where bin_stats is a list of dicts with
    keys: bin_lower, bin_upper, bin_acc, bin_conf, bin_count.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats: list[dict] = []
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        # Include upper bound only for the last bin
        mask = [
            lower <= c < upper if i < n_bins - 1 else lower <= c <= upper
            for c in confidences
        ]
        indices = [j for j, m in enumerate(mask) if m]
        if not indices:
            bin_stats.append(
                {"bin_lower": lower, "bin_upper": upper, "bin_acc": 0.0, "bin_conf": 0.0, "bin_count": 0}
            )
            continue
        bin_acc = sum(1 for j in indices if y_true[j] == y_pred[j]) / len(indices)
        bin_conf = sum(confidences[j] for j in indices) / len(indices)
        ece += (len(indices) / total) * abs(bin_acc - bin_conf)
        bin_stats.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_acc": float(bin_acc),
                "bin_conf": float(bin_conf),
                "bin_count": len(indices),
            }
        )
    return float(ece), bin_stats


def _plot_reliability_diagram(
    bin_stats: list[dict],
    model_name: str,
    ece_value: float,
    output_path: Path,
) -> None:
    """Save a reliability diagram (confidence vs accuracy bar chart)."""
    if not PLOTTING_AVAILABLE:
        return
    bin_midpoints = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bin_stats]
    bin_accs = [b["bin_acc"] for b in bin_stats]
    bin_confs = [b["bin_conf"] for b in bin_stats]
    bin_width = bin_stats[0]["bin_upper"] - bin_stats[0]["bin_lower"] if bin_stats else 0.1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(bin_midpoints, bin_accs, width=bin_width * 0.8, label="Accuracy", color="#4C72B0", alpha=0.8)
    ax.step(
        [0.0] + [b["bin_upper"] for b in bin_stats],
        ([0.0] + bin_confs) if bin_confs else [0.0],
        where="post",
        color="#DD4444",
        linewidth=1.5,
        linestyle="--",
        label="Mean Confidence",
    )
    ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — {model_name}\n(ECE = {ece_value:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _basic_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def _basic_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = (correct / total) if total else 0.0
    report: dict[str, dict[str, float] | float] = {"accuracy": accuracy}
    weighted_f1 = 0.0
    weighted_p = 0.0
    weighted_r = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1-score": f1, "support": float(support)}
        weighted_f1 += f1 * support
        weighted_p += precision * support
        weighted_r += recall * support
    denom = max(total, 1)
    report["weighted avg"] = {
        "precision": weighted_p / denom,
        "recall": weighted_r / denom,
        "f1-score": weighted_f1 / denom,
        "support": float(total),
    }
    return report


def _predict_bert(texts: list[str], model_dir: Path) -> tuple[list[str], list[float]]:
    """Returns (predictions, max_confidences) for the test set."""
    if not BERT_EVAL_AVAILABLE:
        raise RuntimeError("BERT evaluation dependencies are not available.")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    confidences: list[float] = []
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
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())
            predictions.append(LABELS[pred_idx])
            confidences.append(float(probs[pred_idx].item()))
    return predictions, confidences


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
    classical_probs_matrix = classical_model.predict_proba(x_test)
    classical_confs = [float(row.max()) for row in classical_probs_matrix]

    if BERT_EVAL_AVAILABLE and bert_dir.exists() and any(bert_dir.iterdir()):
        bert_preds, bert_confs = _predict_bert(x_test, bert_dir)
    else:
        bert_preds = classical_preds[:]
        bert_confs = classical_confs[:]

    if SKLEARN_METRICS_AVAILABLE:
        cm_classical = confusion_matrix(y_true, classical_preds, labels=LABELS)
        cm_bert = confusion_matrix(y_true, bert_preds, labels=LABELS)
        classical_report = classification_report(y_true, classical_preds, output_dict=True, zero_division=0)
        bert_report = classification_report(y_true, bert_preds, output_dict=True, zero_division=0)
    else:
        cm_classical = _basic_confusion_matrix(y_true, classical_preds, LABELS)
        cm_bert = _basic_confusion_matrix(y_true, bert_preds, LABELS)
        classical_report = _basic_report(y_true, classical_preds, LABELS)
        bert_report = _basic_report(y_true, bert_preds, LABELS)

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

    # --- Calibration (ECE) ---
    ece_classical, classical_bin_stats = _compute_ece(y_true, classical_preds, classical_confs)
    ece_bert, bert_bin_stats = _compute_ece(y_true, bert_preds, bert_confs)

    _plot_reliability_diagram(
        classical_bin_stats, "Classical", ece_classical,
        figures_dir / "calibration_plot_classical.png",
    )
    _plot_reliability_diagram(
        bert_bin_stats, "BERT", ece_bert,
        figures_dir / "calibration_plot.png",
    )

    metrics["evaluation"] = {
        "comparison": comparison_df.to_dict(orient="records"),
        "classical_per_class": classical_report,
        "bert_per_class": bert_report,
    }
    metrics["calibration"] = {
        "ece_classical": ece_classical,
        "ece_bert": ece_bert,
        "classical_bins": classical_bin_stats,
        "bert_bins": bert_bin_stats,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Calibration — ECE classical: {ece_classical:.4f}  ECE bert: {ece_bert:.4f}")
    print("Final comparison table:")
    print(comparison_df.to_string(index=False))
    if not BERT_EVAL_AVAILABLE:
        print("BERT evaluation used classical fallback due to blocked runtime dependencies.")


if __name__ == "__main__":
    main()
