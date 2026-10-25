"""ROC and PR curves, FPR at fixed TPR and confusion matrices from saved probabilities.

Reads the probability files written by src.paper_results (official test split,
reference run of each model) and src.external_eval (external test sets). Nothing
is re-run here.

Binary view everywhere: positive = MALICIOUS, score = P(not-SAFE) = 1 - P(SAFE),
fixed operating point = the API's argmax label (attack unless it says SAFE). On
the official test split the binary view uses the original Hub rows only, as in
the headline table; the 3-class confusion matrix uses all test rows.

Per (evaluation set, model): AUROC, average precision (area under the PR curve),
FPR at TPR = 0.90 and 0.95 (lowest threshold reaching the TPR, so the reported
FPR is the best achievable there), each with a percentile bootstrap 95% CI, and
the confusion counts at the fixed operating point.

Outputs: artifacts/curves.json, artifacts/tables/curves.{md,tex},
artifacts/tables/confusion_binary.{md,tex}, artifacts/tables/confusion_3class.{md,tex},
artifacts/figures/roc_pr.pdf and the plotted points in artifacts/figures/roc_pr_points.csv.

    python -m src.curves
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src import probs_io
from src.metrics import LABELS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TABLES_DIR = ARTIFACTS / "tables"
FIGURES_DIR = ARTIFACTS / "figures"
OUTPUT_PATH = ARTIFACTS / "curves.json"
PAPER_RESULTS = ARTIFACTS / "paper_results.json"

TPR_TARGETS = (0.90, 0.95)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
TEST_SET = "neuralchemy test (original rows)"
# Model names as used in the probability file names, in table order.
MODEL_ORDER = ("classical", "bert", "guard", "protectai", "llama_pg2", "regex", "keyword")
DISPLAY = {
    "classical": "TF-IDF + LR (classical)",
    "bert": "DistilBERT (bert)",
    "guard": "ModernBERT (guard)",
    "protectai": "ProtectAI DeBERTa-v3",
    "llama_pg2": "Llama Prompt Guard 2-86M",
    "regex": "Regex only",
    "keyword": "Keyword list",
}
# Panels of the figure: (evaluation set, title). External sets that were not evaluated are skipped.
FIGURE_SETS = (
    (TEST_SET, "In-distribution"),
    ("qualifire/prompt-injections-benchmark", "Qualifire"),
)


# --------------------------------------------------------------------------- metrics


def binary_targets(labels) -> np.ndarray:
    return np.asarray([str(label) != "SAFE" for label in labels], dtype=bool)


def attack_score(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(probs, dtype=float)[:, LABELS.index("SAFE")]


def fpr_at_tpr(y: np.ndarray, score: np.ndarray, target: float) -> tuple[float, float]:
    """(FPR, threshold) at the highest threshold whose TPR reaches *target*."""
    fpr, tpr, thresholds = roc_curve(y, score)
    idx = int(np.argmax(tpr >= target - 1e-12))
    return float(fpr[idx]), float(thresholds[idx])


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    out = {"auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}
    for target in TPR_TARGETS:
        fpr, threshold = fpr_at_tpr(y, score, target)
        out[f"fpr_at_tpr{int(target * 100)}"] = fpr
        out[f"threshold_at_tpr{int(target * 100)}"] = threshold
    return out


def bootstrap(y: np.ndarray, score: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile 95% CIs of the threshold-free metrics over row resamples (single-class resamples skipped)."""
    rng = np.random.default_rng(seed)
    keys = ["auroc", "auprc"] + [f"fpr_at_tpr{int(t * 100)}" for t in TPR_TARGETS]
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].all() or not y[idx].any():
            continue
        m = point_metrics(y[idx], score[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def confusion_binary(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.asarray(probs).argmax(axis=1) != LABELS.index("SAFE")
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}


def confusion_3class(labels, probs: np.ndarray) -> list[list[int]]:
    """Rows = true label, columns = predicted label, both in LABELS order."""
    true = [LABELS.index(str(label)) for label in labels]
    pred = np.asarray(probs).argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(true, pred):
        matrix[t, p] += 1
    return matrix.tolist()


def evaluate(y: np.ndarray, probs: np.ndarray, n_boot: int) -> dict:
    score = attack_score(probs)
    return {
        "n": int(len(y)),
        "n_attack": int(y.sum()),
        "n_safe": int((~y).sum()),
        **point_metrics(y, score),
        "ci95": bootstrap(y, score, n_boot),
        "fixed_operating_point": confusion_binary(y, probs),
    }


# --------------------------------------------------------------------------- inputs


def test_runs() -> dict[str, Path]:
    """Reference-run probability file per model, as recorded in paper_results.json."""
    results = json.loads(PAPER_RESULTS.read_text(encoding="utf-8"))
    out = {}
    for model, result in results["models"].items():
        ok = [r for r in result["runs"] if r.get("status") == "ok" and r.get("probs_file")]
        if ok:
            out[model] = ROOT / ok[0]["probs_file"]
    return out


def external_runs() -> dict[str, dict[str, Path]]:
    """{dataset slug: {model: file}} from artifacts/probs/external_<slug>_<model>.npz."""
    out: dict[str, dict[str, Path]] = {}
    for path in sorted(probs_io.PROBS_DIR.glob("external_*.npz")):
        stem = path.stem[len("external_"):]
        model = next((m for m in MODEL_ORDER if stem.endswith("_" + m)), None)
        if model is None:
            continue
        out.setdefault(stem[: -len(model) - 1], {})[model] = path
    return out


def unslug(slug: str) -> str:
    return slug.replace("__", "/")


# --------------------------------------------------------------------------- outputs


def _ordered(models) -> list[str]:
    return [m for m in MODEL_ORDER if m in models] + sorted(m for m in models if m not in MODEL_ORDER)


def tables(result: dict) -> dict[str, tuple[list[str], list[list[str]], str]]:
    rows, conf_rows = [], []
    for set_name, models in result["sets"].items():
        for model in _ordered(models):
            m = models[model]
            ci = m["ci95"]
            cell = lambda key, d=3: f"{m[key]:.{d}f} [{ci[key][0]:.{d}f}, {ci[key][1]:.{d}f}]"
            rows.append([set_name, DISPLAY.get(model, model), str(m["n"]), cell("auroc"), cell("auprc"),
                         cell("fpr_at_tpr90"), cell("fpr_at_tpr95")])
            c = m["fixed_operating_point"]
            conf_rows.append([set_name, DISPLAY.get(model, model), str(c["tn"]), str(c["fp"]), str(c["fn"]), str(c["tp"]),
                              f"{c['tpr']:.3f}", f"{c['fpr']:.3f}"])
    out = {
        "curves": (
            ["evaluation set", "model", "n", "AUROC", "AUPRC", "FPR @ TPR=0.90", "FPR @ TPR=0.95"],
            rows,
            "Threshold-free binary metrics (score = 1 - P(SAFE), positive = MALICIOUS) with percentile bootstrap "
            f"95% CIs ({result['config']['n_bootstrap']} resamples). FPR at a fixed TPR uses the highest threshold that "
            "reaches that TPR on the evaluation set itself, so it is an optimistic operating point, not a deployable one.",
        ),
        "confusion_binary": (
            ["evaluation set", "model", "TN", "FP", "FN", "TP", "TPR", "FPR"],
            conf_rows,
            "Binary confusion counts at the fixed operating point the API uses (attack unless the predicted label is SAFE).",
        ),
    }
    three = result.get("test_3class_confusion", {})
    if three:
        out["confusion_3class"] = (
            ["model", "true label"] + [f"pred {label}" for label in LABELS],
            [[DISPLAY.get(model, model), LABELS[i]] + [str(v) for v in row]
             for model in _ordered(three) for i, row in enumerate(three[model])],
            "3-class confusion matrix on all rows of the official test split (rows = true label). SUSPICIOUS is "
            "synthetic; 2-class baselines cannot predict it.",
        )
    return out


def write_figure(curve_data: dict, path: Path) -> list[dict]:
    """ROC and PR panels for each FIGURE_SETS entry present; returns the plotted points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
    panels = [(s, title) for s, title in FIGURE_SETS if s in curve_data]
    fig, axes = plt.subplots(len(panels), 2, figsize=(3.5, 1.6 * len(panels) + 0.55), squeeze=False)
    points = []
    styles = {"classical": "-", "bert": "--", "guard": "-.", "protectai": ":", "llama_pg2": (0, (5, 1, 1, 1))}
    shades = {"classical": "0.0", "bert": "0.0", "guard": "0.0", "protectai": "0.45", "llama_pg2": "0.45"}
    for row, (set_name, title) in enumerate(panels):
        roc_ax, pr_ax = axes[row]
        for model in _ordered(curve_data[set_name]):
            if model not in styles:
                continue  # discrete baselines are single points; they are in the table
            y, score = curve_data[set_name][model]
            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)
            label = DISPLAY.get(model, model)
            roc_ax.plot(fpr, tpr, linestyle=styles[model], color=shades[model], linewidth=0.9, label=label)
            pr_ax.plot(recall, precision, linestyle=styles[model], color=shades[model], linewidth=0.9, label=label)
            points += [{"set": set_name, "model": model, "curve": "roc", "x": float(a), "y": float(b)} for a, b in zip(fpr, tpr)]
            points += [{"set": set_name, "model": model, "curve": "pr", "x": float(a), "y": float(b)}
                       for a, b in zip(recall, precision)]
        roc_ax.plot([0, 1], [0, 1], color="0.7", linewidth=0.5)
        roc_ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate", ylabel="True-positive rate")
        pr_ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
        roc_ax.set_title(f"{title}: ROC", fontsize=7)
        pr_ax.set_title(f"{title}: PR", fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=5.5, frameon=False, handlelength=3)
    fig.tight_layout(pad=0.4, rect=(0.03, 0.13, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)  # preview for checking; the paper uses the vector PDF
    plt.close(fig)
    return points


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args(argv)

    from src.paper_results import to_latex, to_markdown
    from src.preprocess import get_splits

    _, _, test_df, _ = get_splits()
    original = (test_df["derivation"] == "original").to_numpy()
    sets: dict[str, dict] = {TEST_SET: {}}
    curve_data: dict[str, dict] = {TEST_SET: {}}
    three_class: dict[str, list] = {}
    for model, path in test_runs().items():
        saved = probs_io.load_probs(path, test_df["text"].tolist())
        probs, labels = saved["probs"], saved["labels"]
        y = binary_targets(labels[original])
        sets[TEST_SET][model] = {"probs_file": path.relative_to(ROOT).as_posix(), **evaluate(y, probs[original], args.bootstrap)}
        curve_data[TEST_SET][model] = (y, attack_score(probs[original]))
        three_class[model] = confusion_3class(labels, probs)
    for slug, models in external_runs().items():
        name = unslug(slug)
        sets[name], curve_data[name] = {}, {}
        for model, path in models.items():
            saved = probs_io.load_probs(path)
            y = binary_targets(saved["labels"])
            sets[name][model] = {"probs_file": path.relative_to(ROOT).as_posix(), **evaluate(y, saved["probs"], args.bootstrap)}
            curve_data[name][model] = (y, attack_score(saved["probs"]))

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"n_bootstrap": args.bootstrap, "bootstrap_seed": BOOTSTRAP_SEED, "tpr_targets": list(TPR_TARGETS),
                   "score": "1 - P(SAFE)", "positive": "MALICIOUS (any non-SAFE true label)",
                   "test_rows": "original Hub rows of the official test split"},
        "sets": sets,
        "test_3class_confusion": three_class,
    }
    points = write_figure(curve_data, FIGURES_DIR / "roc_pr.pdf")
    pd.DataFrame(points).to_csv(FIGURES_DIR / "roc_pr_points.csv", index=False)
    result["figure"] = {"pdf": "artifacts/figures/roc_pr.pdf", "points_csv": "artifacts/figures/roc_pr_points.csv",
                        "panels": [s for s, _ in FIGURE_SETS if s in curve_data]}
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, (columns, rows, caption) in tables(result).items():
        (TABLES_DIR / f"{name}.md").write_text(f"{caption}\n\n" + to_markdown(columns, rows), encoding="utf-8")
        (TABLES_DIR / f"{name}.tex").write_text(to_latex(columns, rows, caption, f"tab:{name.replace('_', '-')}"),
                                                encoding="utf-8")
    print(to_markdown(*tables(result)["curves"][:2]))
    return result


if __name__ == "__main__":
    main()
