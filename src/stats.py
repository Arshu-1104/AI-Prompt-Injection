"""Bootstrap confidence intervals and paired significance tests.

Resampling is over the rows of the fixed official test split (seed variation is
reported separately). Metrics are computed with vectorized numpy so 1000
resamples stay fast; tests check them against sklearn / src.metrics.
"""
from __future__ import annotations

import math

import numpy as np

from src.metrics import LABELS

N_BOOTSTRAP = 1000
ECE_BINS = 10


def _ece(correct: np.ndarray, conf: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Equal-width-bin ECE, same binning as src.metrics.compute_ece."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # [lower, upper) bins, last bin closed - identical to compute_ece's edges.
    bins = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    total = len(conf)
    counts = np.bincount(bins, minlength=n_bins)
    acc = np.bincount(bins, weights=correct, minlength=n_bins)
    cnf = np.bincount(bins, weights=conf, minlength=n_bins)
    nz = counts > 0
    return float(np.sum(np.abs(acc[nz] - cnf[nz])) / total)


def _f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    tp = np.bincount(y_true[y_true == y_pred], minlength=k).astype(float)
    pred_count = np.bincount(y_pred, minlength=k).astype(float)
    support = np.bincount(y_true, minlength=k).astype(float)
    denom = pred_count + support
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f1, support


def three_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray) -> dict:
    """y_true / y_pred are class indices into LABELS."""
    k = len(LABELS)
    correct = (y_true == y_pred).astype(float)
    f1, support = _f1_per_class(y_true, y_pred, k)
    out = {
        "accuracy": float(correct.mean()),
        "f1_weighted": float(np.sum(f1 * support) / support.sum()),
        "ece": _ece(correct, conf),
    }
    out.update({f"f1_{label}": float(f1[i]) for i, label in enumerate(LABELS)})
    return out


def binary_metrics(y_true_attack: np.ndarray, pred_attack: np.ndarray, conf: np.ndarray) -> dict:
    """Booleans: True = MALICIOUS. F1 is for the MALICIOUS class."""
    correct = (y_true_attack == pred_attack).astype(float)
    tp = float(np.sum(y_true_attack & pred_attack))
    fp = float(np.sum(~y_true_attack & pred_attack))
    fn = float(np.sum(y_true_attack & ~pred_attack))
    negatives = float(np.sum(~y_true_attack))
    return {
        "accuracy": float(correct.mean()),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "false_positive_rate": fp / negatives if negatives else float("nan"),
        "ece": _ece(correct, conf),
    }


class Evaluated:
    """Per-row predictions of one model on one test split, ready for resampling."""

    def __init__(self, labels: list[str], derivation: list[str], probs: np.ndarray):
        probs = np.asarray(probs, dtype=float)
        self.y_true = np.array([LABELS.index(label) for label in labels])
        self.y_pred = probs.argmax(axis=1)
        self.conf = probs.max(axis=1)
        self.original = np.array([d == "original" for d in derivation])
        safe = LABELS.index("SAFE")
        self.bin_true = self.y_true[self.original] != safe
        self.bin_pred = self.y_pred[self.original] != safe
        p_safe = probs[self.original, safe]
        self.bin_conf = np.where(self.bin_pred, 1.0 - p_safe, p_safe)

    def correct_3c(self) -> np.ndarray:
        return self.y_true == self.y_pred

    def correct_bin(self) -> np.ndarray:
        return self.bin_true == self.bin_pred

    def metrics(self, idx3: np.ndarray | None = None, idxb: np.ndarray | None = None) -> dict:
        idx3 = np.arange(len(self.y_true)) if idx3 is None else idx3
        idxb = np.arange(len(self.bin_true)) if idxb is None else idxb
        return {
            "three_class": three_class_metrics(self.y_true[idx3], self.y_pred[idx3], self.conf[idx3]),
            "binary": binary_metrics(self.bin_true[idxb], self.bin_pred[idxb], self.bin_conf[idxb]),
        }


def resample_indices(n3: int, nb: int, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Shared resample indices so every model (and every pair) sees the same resamples."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n3, size=(n_boot, n3)), rng.integers(0, nb, size=(n_boot, nb))


def bootstrap_ci(ev: Evaluated, idx3: np.ndarray, idxb: np.ndarray) -> dict:
    """Point estimate plus percentile 95% CI for every metric."""
    point = ev.metrics()
    samples = [ev.metrics(i3, ib) for i3, ib in zip(idx3, idxb)]
    out: dict[str, dict] = {}
    for view in ("three_class", "binary"):
        out[view] = {}
        for metric, value in point[view].items():
            values = np.array([s[view][metric] for s in samples], dtype=float)
            values = values[~np.isnan(values)]
            lo, hi = np.percentile(values, [2.5, 97.5]) if len(values) else (float("nan"), float("nan"))
            out[view][metric] = {"point": value, "ci95": [float(lo), float(hi)]}
    out["n_bootstrap"] = int(len(idx3))
    return out


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired correctness."""
    from scipy.stats import binom

    b = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(b, c), n, 0.5)))
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def paired_comparison(a: Evaluated, b: Evaluated, idx3: np.ndarray, idxb: np.ndarray) -> dict:
    """McNemar plus paired-bootstrap CI of the accuracy difference (A - B), both views."""
    if len(a.y_true) != len(b.y_true) or not np.array_equal(a.y_true, b.y_true):
        raise ValueError("Paired comparison needs predictions on the same rows.")
    out = {}
    for view, ca, cb, idx in (
        ("three_class", a.correct_3c(), b.correct_3c(), idx3),
        ("binary", a.correct_bin(), b.correct_bin(), idxb),
    ):
        diffs = ca[idx].mean(axis=1) - cb[idx].mean(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[view] = {
            "accuracy_a": float(ca.mean()),
            "accuracy_b": float(cb.mean()),
            "difference": float(ca.mean() - cb.mean()),
            "paired_bootstrap_ci95": [float(lo), float(hi)],
            "mcnemar": mcnemar_exact(ca, cb),
        }
    return out


def fmt_ci(entry: dict | None, digits: int = 4) -> str:
    if not entry or entry.get("point") is None or (isinstance(entry["point"], float) and math.isnan(entry["point"])):
        return "-"
    lo, hi = entry["ci95"]
    return f"{entry['point']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
