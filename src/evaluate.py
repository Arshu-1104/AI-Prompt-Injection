"""Evaluate classical, bert and guard on the official test split.

    python -m src.evaluate                      # all three models, CPU latency
    python -m src.evaluate --models classical bert --max-samples 200

Reports two views per model (see src.metrics.evaluate_split):
* binary  - SAFE vs MALICIOUS on the original Hub test rows (headline)
* 3-class - SAFE / SUSPICIOUS / MALICIOUS on all test rows; SUSPICIOUS is synthetic
Models that cannot be loaded are reported as skipped with the reason.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import LABELS, compute_ece, evaluate_split, labels_from_probs
from src.preprocess import get_splits

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    PLOTTING_AVAILABLE = True
except Exception as exc:
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)


MODEL_TYPES = ["classical", "bert", "guard"]
WARMUP_CALLS = 3
LATENCY_SAMPLES = 200

# Kept for backwards compatibility with older imports.
_compute_ece = compute_ece


def _plot_reliability_diagram(bin_stats: list[dict], model_name: str, ece_value: float, output_path: Path) -> None:
    """Save a reliability diagram (confidence vs accuracy bar chart)."""
    if not PLOTTING_AVAILABLE:
        return
    bin_midpoints = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bin_stats]
    bin_width = bin_stats[0]["bin_upper"] - bin_stats[0]["bin_lower"] if bin_stats else 0.1
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(bin_midpoints, [b["bin_acc"] for b in bin_stats], width=bin_width * 0.8, label="Accuracy", color="#4C72B0", alpha=0.8)
    ax.step(
        [0.0] + [b["bin_upper"] for b in bin_stats],
        [0.0] + [b["bin_conf"] for b in bin_stats],
        where="post", color="#DD4444", linewidth=1.5, linestyle="--", label="Mean Confidence",
    )
    ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — {model_name} (3-class)\n(ECE = {ece_value:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _evaluate_one(model_type: str, test_df: pd.DataFrame, device: str) -> tuple[dict, np.ndarray]:
    from src.predict import PromptAnalyzer

    analyzer = PromptAnalyzer(model_type=model_type, device=device)
    texts = test_df["text"].tolist()
    probs = np.concatenate([analyzer.predict_proba(texts[i : i + 64]) for i in range(0, len(texts), 64)])

    sample = texts[:LATENCY_SAMPLES]
    for text in sample[:WARMUP_CALLS]:
        analyzer.predict_proba([text])
    latencies_ms = []
    for text in sample:  # one prompt at a time, as the API serves them
        t0 = time.perf_counter()
        analyzer.predict_proba([text])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    summary = evaluate_split(test_df, probs)
    summary.update(
        {
            "model": model_type,
            "status": "ok",
            "source": analyzer.source,
            "device": device if model_type != "classical" else "cpu",
            "latency_samples": len(latencies_ms),
            "mean_latency_ms": float(statistics.fmean(latencies_ms)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        }
    )
    return summary, probs


def evaluate_all(
    models: list[str] | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
    write_artifacts: bool = True,
) -> list[dict]:
    """Evaluate each model; returns one row per model (status 'ok' or 'skipped')."""
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    figures_dir = artifacts_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_df, provenance = get_splits()
    if max_samples:
        test_df = test_df.sample(n=min(max_samples, len(test_df)), random_state=42)
        provenance["evaluated_subset"] = {"max_samples": max_samples, "rows": len(test_df)}

    rows: list[dict] = []
    outputs: dict[str, np.ndarray] = {}
    for model_type in models or MODEL_TYPES:
        print(f"Evaluating {model_type} on {len(test_df)} test rows ...")
        try:
            summary, probs = _evaluate_one(model_type, test_df, device)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  skipped: {reason}")
            rows.append({"model": model_type, "status": "skipped", "reason": reason})
            continue
        rows.append(summary)
        outputs[model_type] = probs

    if not outputs:
        raise RuntimeError("No model could be evaluated; see the skip reasons above.")
    if write_artifacts:
        _write_artifacts(rows, outputs, test_df, artifacts_dir, figures_dir, provenance)
    return rows


def _write_artifacts(rows, outputs, test_df, artifacts_dir: Path, figures_dir: Path, provenance: dict) -> None:
    y_true = test_df["label"].tolist()
    for model_type, probs in outputs.items():
        preds = labels_from_probs(probs)
        ece, bins = compute_ece(y_true, preds, probs.max(axis=1).tolist())
        _plot_reliability_diagram(bins, model_type, ece, figures_dir / f"calibration_plot_{model_type}.png")
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_true, preds, labels=LABELS), annot=True, fmt="d", cmap="Blues",
                        xticklabels=LABELS, yticklabels=LABELS)
            plt.title(f"Confusion Matrix - {model_type} (3-class; SUSPICIOUS is synthetic)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(figures_dir / f"confusion_matrix_{model_type}.png", dpi=140)
            plt.close()
    if not PLOTTING_AVAILABLE:
        print(f"Plotting unavailable: {PLOTTING_IMPORT_ERROR}")

    table = comparison_table(rows)
    metrics_path = artifacts_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["evaluation"] = {"models": rows, "dataset": provenance}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison = {"dataset": provenance, "rows": table.astype(object).where(table.notna(), None).to_dict(orient="records")}
    (artifacts_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Binary (headline) and 3-class metrics plus mean CPU latency per model."""
    table = []
    for r in rows:
        if r["status"] != "ok":
            table.append({"model": r["model"], "status": "skipped"})
            continue
        b, t = r["binary"], r["three_class"]
        table.append(
            {
                "model": r["model"],
                "status": "ok",
                "bin_accuracy": round(b["accuracy"], 4),
                "bin_f1": round(b["f1"], 4),
                "bin_fpr": None if b["false_positive_rate"] is None else round(b["false_positive_rate"], 4),
                "bin_ece": round(b["ece"], 4),
                "3c_accuracy": round(t["accuracy"], 4),
                "3c_f1_weighted": round(t["f1_weighted"], 4),
                "3c_ece": round(t["ece"], 4),
                "mean_latency_ms": round(r["mean_latency_ms"], 2),
                "n_binary": b["n_samples"],
                "n_3class": t["n_samples"],
            }
        )
    return pd.DataFrame(table)


def print_comparison(rows: list[dict]) -> None:
    print("\nModel comparison on the official test split")
    print("  bin_* = SAFE vs MALICIOUS on original Hub rows (headline); 3c_* = 3-class incl. synthetic SUSPICIOUS")
    print(comparison_table(rows).to_string(index=False, na_rep="-"))
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['model']} skipped: {r['reason']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    parser.add_argument("--device", default="cpu", help="Torch device for bert/guard (default: cpu, so latency is CPU latency).")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate on a random subset of the test split.")
    args = parser.parse_args(argv)
    print_comparison(evaluate_all(args.models, device=args.device, max_samples=args.max_samples))


if __name__ == "__main__":
    main()
