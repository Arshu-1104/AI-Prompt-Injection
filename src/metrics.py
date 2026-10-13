"""Classification and calibration metrics shared by training and evaluation."""
from __future__ import annotations

import numpy as np

LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]


def compute_ece(
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
                {"bin_lower": float(lower), "bin_upper": float(upper), "bin_acc": 0.0, "bin_conf": 0.0, "bin_count": 0}
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


def summarize_predictions(y_true: list[str], y_pred: list[str], confidences: list[float]) -> dict:
    """Accuracy, weighted precision/recall/F1, per-class F1 and ECE."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    ece, _ = compute_ece(y_true, y_pred, confidences)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "per_class_f1": {label: float(score) for label, score in zip(LABELS, per_class)},
        "ece": ece,
        "n_samples": len(y_true),
    }
