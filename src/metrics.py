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


# --------------------------------------------------------------------------- probabilities


def labels_from_probs(probs: np.ndarray) -> list[str]:
    """Argmax label per row of an (n, 3) matrix whose columns follow LABELS."""
    return [LABELS[i] for i in np.asarray(probs).argmax(axis=1)]


def summarize_binary(y_true: list[str], probs: np.ndarray) -> dict:
    """SAFE vs MALICIOUS metrics from 3-class probabilities.

    *y_true* holds "SAFE"/"MALICIOUS". A row is predicted SAFE exactly when the
    3-class argmax is SAFE (the label the API returns); otherwise MALICIOUS.
    Confidence is P(SAFE) for SAFE predictions and 1 - P(SAFE) otherwise.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    probs = np.asarray(probs, dtype=float)
    pred_safe = probs.argmax(axis=1) == LABELS.index("SAFE")
    y_pred = ["SAFE" if s else "MALICIOUS" for s in pred_safe]
    confidences = np.where(pred_safe, probs[:, 0], 1.0 - probs[:, 0]).tolist()
    negatives = [t == "SAFE" for t in y_true]
    false_pos = sum(1 for t, p in zip(y_true, y_pred) if t == "SAFE" and p == "MALICIOUS")
    ece, _ = compute_ece(list(y_true), y_pred, confidences)
    kwargs = {"pos_label": "MALICIOUS", "zero_division": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, **kwargs)),
        "false_positive_rate": float(false_pos / sum(negatives)) if sum(negatives) else None,
        "ece": ece,
        "n_samples": len(y_true),
    }


def evaluate_split(df, probs: np.ndarray) -> dict:
    """Both views of one evaluated split.

    three_class: all rows, including synthetic SUSPICIOUS and other derived rows.
    binary: original Hub rows only (derivation == "original"), Hub labels -
    the headline for comparison with other work.
    """
    probs = np.asarray(probs, dtype=float)
    three_class = summarize_predictions(df["label"].tolist(), labels_from_probs(probs), probs.max(axis=1).tolist())
    original = (df["derivation"] == "original").to_numpy()
    binary = summarize_binary(df.loc[original, "label"].tolist(), probs[original])
    return {"three_class": three_class, "binary": binary}


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Two-sided 95% Wilson score interval for a binomial proportion (None if n == 0)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_labels(df, y_pred: list[str]) -> dict:
    """Label-only metrics (no ECE) for predictions that are not probabilities, e.g.
    the API pipeline after the rule layer. Same two views as evaluate_split."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true = df["label"].tolist()
    original = (df["derivation"] == "original").to_numpy()
    bin_true = [t for t, o in zip(y_true, original) if o]
    bin_pred = ["SAFE" if p == "SAFE" else "MALICIOUS" for p, o in zip(y_pred, original) if o]
    negatives = sum(t == "SAFE" for t in bin_true)
    false_pos = sum(1 for t, p in zip(bin_true, bin_pred) if t == "SAFE" and p == "MALICIOUS")
    return {
        "three_class": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        },
        "binary": {
            "accuracy": float(accuracy_score(bin_true, bin_pred)),
            "f1": float(f1_score(bin_true, bin_pred, pos_label="MALICIOUS", zero_division=0)),
            "false_positive_rate": float(false_pos / negatives) if negatives else None,
        },
    }
