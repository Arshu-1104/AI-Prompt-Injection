"""Evaluate classical, bert and guard on the held-out test set.

    python -m src.evaluate                      # all three models, CPU latency
    python -m src.evaluate --models classical bert --max-samples 200

Models that cannot be loaded (no torch, no weights, ...) are reported as
skipped with the reason instead of being silently replaced.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import confusion_matrix

    SKLEARN_METRICS_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False

from src.metrics import LABELS, compute_ece, summarize_predictions
from src.preprocess import get_splits_with_provenance

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception as exc:
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)


MODEL_TYPES = ["classical", "bert", "guard"]
WARMUP_CALLS = 3

# Kept for backwards compatibility with older imports.
_compute_ece = compute_ece


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


def _evaluate_one(model_type: str, texts: list[str], y_true: list[str], device: str) -> tuple[dict, list[str], list[float]]:
    """Predict every test prompt one at a time (as the API does) and time each call."""
    from src.predict import PromptAnalyzer

    analyzer = PromptAnalyzer(model_type=model_type, device=device)
    predict_fn = {
        "classical": analyzer._predict_classical,
        "bert": analyzer._predict_bert,
        "guard": analyzer._predict_guard,
    }[model_type]
    for text in texts[:WARMUP_CALLS]:
        predict_fn(text)

    preds: list[str] = []
    confs: list[float] = []
    latencies_ms: list[float] = []
    for text in texts:
        t0 = time.perf_counter()
        label, confidence = predict_fn(text)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        preds.append(label)
        confs.append(confidence)

    summary = summarize_predictions(y_true, preds, confs)
    summary.update(
        {
            "model": model_type,
            "status": "ok",
            "source": analyzer.source,
            "device": device if model_type != "classical" else "cpu",
            "mean_latency_ms": float(statistics.fmean(latencies_ms)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        }
    )
    return summary, preds, confs


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

    _, test_df, provenance = get_splits_with_provenance()
    if max_samples:
        test_df = test_df.sample(n=min(max_samples, len(test_df)), random_state=42)
        provenance["evaluated_subset"] = {"max_samples": max_samples, "rows": len(test_df)}
    x_test = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    rows: list[dict] = []
    outputs: dict[str, tuple[list[str], list[float]]] = {}
    for model_type in models or MODEL_TYPES:
        print(f"Evaluating {model_type} on {len(x_test)} prompts ...")
        try:
            summary, preds, confs = _evaluate_one(model_type, x_test, y_true, device)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  skipped: {reason}")
            rows.append({"model": model_type, "status": "skipped", "reason": reason})
            continue
        rows.append(summary)
        outputs[model_type] = (preds, confs)

    if not outputs:
        raise RuntimeError("No model could be evaluated; see the skip reasons above.")

    if write_artifacts:
        _write_artifacts(rows, outputs, y_true, artifacts_dir, figures_dir, provenance)
    return rows


def _write_artifacts(
    rows: list[dict],
    outputs: dict[str, tuple[list[str], list[float]]],
    y_true: list[str],
    artifacts_dir: Path,
    figures_dir: Path,
    provenance: dict,
) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    calibration: dict = {}
    for model_type, (preds, confs) in outputs.items():
        ece, bins = compute_ece(y_true, preds, confs)
        calibration[f"ece_{model_type}"] = ece
        calibration[f"{model_type}_bins"] = bins
        _plot_reliability_diagram(bins, model_type, ece, figures_dir / f"calibration_plot_{model_type}.png")
        if PLOTTING_AVAILABLE:
            cm = (
                confusion_matrix(y_true, preds, labels=LABELS)
                if SKLEARN_METRICS_AVAILABLE
                else _basic_confusion_matrix(y_true, preds, LABELS)
            )
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
            plt.title(f"Confusion Matrix - {model_type}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(figures_dir / f"confusion_matrix_{model_type}.png", dpi=140)
            plt.close()

    comparison_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "precision_weighted": r["precision_weighted"],
                "recall_weighted": r["recall_weighted"],
                "f1_weighted": r["f1_weighted"],
            }
            for r in ok_rows
        ]
    )
    if PLOTTING_AVAILABLE:
        melted = comparison_df.melt(id_vars="model", var_name="metric", value_name="score")
        plt.figure(figsize=(10, 5))
        sns.barplot(data=melted, x="metric", y="score", hue="model")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=20)
        plt.title("Model Comparison on Held-Out Test Set")
        plt.tight_layout()
        plt.savefig(figures_dir / "model_comparison.png", dpi=140)
        plt.close()

        per_class_df = pd.DataFrame(
            [{"class": label, "model": r["model"], "f1": r["per_class_f1"][label]} for r in ok_rows for label in LABELS]
        )
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
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["evaluation"] = {
        "comparison": comparison_df.to_dict(orient="records"),
        "models": rows,
        "dataset": provenance,
    }
    metrics["calibration"] = calibration
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    comparison_table(rows).to_csv(artifacts_dir / "model_comparison.csv", index=False)


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Accuracy, weighted F1, ECE and mean CPU latency per model."""
    table = []
    for r in rows:
        if r["status"] != "ok":
            table.append({"model": r["model"], "status": "skipped", "accuracy": None, "f1_weighted": None,
                          "ece": None, "mean_latency_ms": None, "n": None})
            continue
        table.append(
            {
                "model": r["model"],
                "status": "ok",
                "accuracy": round(r["accuracy"], 4),
                "f1_weighted": round(r["f1_weighted"], 4),
                "ece": round(r["ece"], 4),
                "mean_latency_ms": round(r["mean_latency_ms"], 2),
                "n": r["n_samples"],
            }
        )
    df = pd.DataFrame(table)
    df["n"] = df["n"].astype("Int64")
    return df


def print_comparison(rows: list[dict]) -> None:
    print("\nModel comparison (held-out test set, latency = single prompt):")
    print(comparison_table(rows).to_string(index=False, na_rep="-"))
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['model']} skipped: {r['reason']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    parser.add_argument("--device", default="cpu", help="Torch device for bert/guard (default: cpu, so latency is CPU latency).")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate on a random subset of the test set.")
    args = parser.parse_args(argv)
    rows = evaluate_all(args.models, device=args.device, max_samples=args.max_samples)
    print_comparison(rows)


if __name__ == "__main__":
    main()
