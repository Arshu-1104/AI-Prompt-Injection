"""Paper results: every number in the paper's tables is computed here.

For each model (classical, bert, guard) and each training seed, the model is
trained from scratch on the canonical split (src.load_data / src.preprocess)
and evaluated on CPU on the same held-out test set:

* accuracy, weighted F1, per-class F1, ECE (3-class)
* SAFE-vs-attack binary accuracy / F1 (attack = SUSPICIOUS or MALICIOUS)
* per-category robustness: the adversarial suite in src/robustness.py and the
  dataset's own attack categories (rows = category, columns = model)
* mean / p95 single-prompt CPU latency (first seed's model)

Seeds are aggregated as mean +/- sample std with a two-sided 95% Student-t
confidence interval. Optionally the public baseline
protectai/deberta-v3-base-prompt-injection-v2 is evaluated as-is (no training,
one run); its INJECTION label is mapped to MALICIOUS, so it can never predict
SUSPICIOUS and the binary columns are the fair comparison.

Outputs: artifacts/paper_results.json and artifacts/tables/*.md|*.tex.
Per-seed weights go to artifacts/paper_runs/ (models/ is never touched).

    python -m src.paper_results                         # 5 seeds classical+bert, 1 guard
    python -m src.paper_results --guard-seeds 3 --protectai
    python -m src.paper_results --max-samples 64 --epochs 1   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import LABELS, summarize_predictions
from src.preprocess import clean_text, get_splits_with_provenance
from src.robustness import ADVERSARIAL_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
TABLES_DIR = ARTIFACTS_DIR / "tables"
RUNS_DIR = ARTIFACTS_DIR / "paper_runs"
RESULTS_PATH = ARTIFACTS_DIR / "paper_results.json"

PROTECTAI_ID = "protectai/deberta-v3-base-prompt-injection-v2"
DISPLAY_NAMES = {
    "classical": "TF-IDF + linear (classical)",
    "bert": "DistilBERT (bert)",
    "guard": "ModernBERT-base (guard)",
    "protectai": "ProtectAI DeBERTa-v3 (baseline)",
}
WARMUP_CALLS = 3
MIN_CATEGORY_ROWS = 10  # smaller dataset categories are pooled into one row


# --------------------------------------------------------------------------- statistics


def summarize_values(values: list[float]) -> dict:
    """Mean, sample std and two-sided 95% Student-t CI over seeds (None for n < 2)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = float(statistics.fmean(values))
    if n < 2:
        return {"n": 1, "mean": mean, "std": None, "ci95": None}
    from scipy.stats import t

    std = float(statistics.stdev(values))
    half = float(t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def to_binary(label: str) -> str:
    return "SAFE" if label == "SAFE" else "ATTACK"


# --------------------------------------------------------------------------- predictors


class ClassicalPredictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.classes = [str(c) for c in pipeline.classes_]

    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        probs = self.pipeline.predict_proba(texts)
        idx = probs.argmax(axis=1)
        return [self.classes[i] for i in idx], probs.max(axis=1).astype(float).tolist()


class TransformerPredictor:
    """Any sequence classifier on CPU; label names are mapped onto LABELS."""

    def __init__(self, source: str, max_length: int = 256, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source).to("cpu").eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.labels = [self._map_label(self.model.config.id2label[i]) for i in range(self.model.config.num_labels)]

    @staticmethod
    def _map_label(name: str) -> str:
        upper = str(name).upper()
        if upper in LABELS:
            return upper
        if "INJECTION" in upper or "JAILBREAK" in upper or "UNSAFE" in upper:
            return "MALICIOUS"
        if upper in {"BENIGN", "LEGIT", "LEGITIMATE"}:
            return "SAFE"
        raise ValueError(f"Cannot map model label {name!r} onto {LABELS}")

    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        labels: list[str] = []
        confs: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start : start + self.batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                probs = self.torch.softmax(self.model(**batch).logits.float(), dim=-1).numpy()
                labels += [self.labels[i] for i in probs.argmax(axis=1)]
                confs += probs.max(axis=1).astype(float).tolist()
        return labels, confs


# --------------------------------------------------------------------------- evaluation


def _category_frame(test_df: pd.DataFrame) -> pd.Series:
    """Dataset category per test row, with rare categories pooled."""
    counts = test_df["category"].value_counts()
    rare = set(counts[counts < MIN_CATEGORY_ROWS].index)
    pooled = f"other (<{MIN_CATEGORY_ROWS} rows each)"
    return test_df["category"].map(lambda c: pooled if c in rare else c)


def evaluate_predictor(predictor, test_df: pd.DataFrame, categories: pd.Series) -> dict:
    y_true = test_df["label"].tolist()
    y_pred, confs = predictor.predict(test_df["text"].tolist())
    from sklearn.metrics import accuracy_score, f1_score

    result = summarize_predictions(y_true, y_pred, confs)
    true_bin = [to_binary(label) for label in y_true]
    pred_bin = [to_binary(label) for label in y_pred]
    result["binary_accuracy"] = float(accuracy_score(true_bin, pred_bin))
    result["binary_f1"] = float(f1_score(true_bin, pred_bin, pos_label="ATTACK", zero_division=0))

    correct = pd.Series([t == p for t, p in zip(true_bin, pred_bin)], index=test_df.index)
    result["dataset_categories"] = {
        str(cat): {"n": int(mask.sum()), "detection_accuracy": float(correct[mask].mean())}
        for cat, mask in ((c, categories == c) for c in sorted(categories.unique()))
    }

    adv_texts = [clean_text(item["text"]) for item in ADVERSARIAL_PROMPTS]  # same normalization as the API
    adv_pred, _ = predictor.predict(adv_texts)
    adversarial: dict[str, dict] = {}
    for item, pred in zip(ADVERSARIAL_PROMPTS, adv_pred):
        entry = adversarial.setdefault(item["category"], {"n": 0, "binary_correct": 0, "exact_correct": 0})
        entry["n"] += 1
        entry["binary_correct"] += int(to_binary(pred) == to_binary(item["expected"]))
        entry["exact_correct"] += int(pred == item["expected"])
    result["adversarial_categories"] = {
        cat: {
            "n": e["n"],
            "detection_accuracy": e["binary_correct"] / e["n"],
            "exact_accuracy": e["exact_correct"] / e["n"],
        }
        for cat, e in sorted(adversarial.items())
    }
    return result


def measure_latency(predictor, texts: list[str]) -> dict:
    for text in texts[:WARMUP_CALLS]:
        predictor.predict([text])
    timings = []
    for text in texts:
        t0 = time.perf_counter()
        predictor.predict([text])
        timings.append((time.perf_counter() - t0) * 1000)
    return {
        "n": len(timings),
        "mean_ms": float(statistics.fmean(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


# --------------------------------------------------------------------------- per-model runs


def _seed_list(base: int, n: int) -> list[int]:
    return [base + i for i in range(n)]


def run_classical(seeds, test_df, categories, latency_texts) -> dict:
    from src.train_classical import fit_classical

    runs, latency = [], None
    for seed in seeds:
        print(f"[classical] seed {seed}: fitting (5-fold CV model selection) ...")
        fitted = fit_classical(random_state=seed)
        predictor = ClassicalPredictor(fitted["pipeline"])
        metrics = evaluate_predictor(predictor, test_df, categories)
        runs.append({"seed": seed, "status": "ok", "selected_model": fitted["best_name"], **metrics})
        if latency is None:
            latency = measure_latency(predictor, latency_texts)
    return {"runs": runs, "latency": latency}


def run_transformer(model_type, seeds, test_df, categories, latency_texts, args) -> dict:
    from src.transformer_training import TrainingError, build_arg_parser, train_model

    spec = __import__(
        {"bert": "src.train_bert", "guard": "src.train_guard_model"}[model_type], fromlist=["SPEC"]
    ).SPEC
    runs, latency = [], None
    for seed in seeds:
        out_dir = RUNS_DIR / model_type / f"seed_{seed}"
        cli = [
            "--seed", str(seed), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--output-dir", str(out_dir),
        ]
        if args.max_samples:
            cli += ["--max-samples", str(args.max_samples)]
        print(f"[{model_type}] seed {seed}: training -> {out_dir}")
        try:
            train_model(spec, build_arg_parser(spec).parse_args(cli), shared_outputs=False)
            predictor = TransformerPredictor(str(out_dir))
        except Exception as exc:  # record the failure and keep going
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[{model_type}] seed {seed} FAILED: {reason}")
            runs.append({"seed": seed, "status": "failed", "reason": reason})
            if isinstance(exc, TrainingError) and "could not be imported" in str(exc):
                for rest in seeds[seeds.index(seed) + 1 :]:
                    runs.append({"seed": rest, "status": "failed", "reason": reason})
                break
            continue
        metrics = evaluate_predictor(predictor, test_df, categories)
        runs.append({"seed": seed, "status": "ok", "weights": out_dir.relative_to(ROOT).as_posix(), **metrics})
        if latency is None:
            latency = measure_latency(predictor, latency_texts)
    return {"runs": runs, "latency": latency}


def run_protectai(test_df, categories, latency_texts) -> dict:
    print(f"[protectai] evaluating {PROTECTAI_ID} (no training) ...")
    try:
        predictor = TransformerPredictor(PROTECTAI_ID, max_length=512)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"[protectai] FAILED: {reason}")
        return {"runs": [{"seed": None, "status": "failed", "reason": reason}], "latency": None}
    metrics = evaluate_predictor(predictor, test_df, categories)
    return {
        "runs": [{"seed": None, "status": "ok", **metrics}],
        "latency": measure_latency(predictor, latency_texts),
        "label_mapping": {"SAFE": "SAFE", "INJECTION": "MALICIOUS"},
    }


def aggregate(model_result: dict) -> dict:
    ok = [r for r in model_result["runs"] if r["status"] == "ok"]
    summary: dict = {
        "n_seeds_ok": len(ok),
        "n_seeds_failed": len(model_result["runs"]) - len(ok),
    }
    for metric in ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "ece", "binary_accuracy", "binary_f1"):
        summary[metric] = summarize_values([r[metric] for r in ok])
    summary["per_class_f1"] = {label: summarize_values([r["per_class_f1"][label] for r in ok]) for label in LABELS}
    for key in ("adversarial_categories", "dataset_categories"):
        cats = sorted({c for r in ok for c in r[key]})
        summary[key] = {
            c: {
                "n": ok[0][key][c]["n"],
                "detection_accuracy": summarize_values([r[key][c]["detection_accuracy"] for r in ok]),
            }
            for c in cats
        }
    return summary


# --------------------------------------------------------------------------- tables


def fmt_stat(stat: dict | None, digits: int = 4) -> str:
    if not stat or stat.get("mean") is None:
        return "-"
    text = f"{stat['mean']:.{digits}f}"
    if stat.get("std") is not None:
        lo, hi = stat["ci95"]
        text += f" ± {stat['std']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return text


def to_markdown(columns: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def to_latex(columns: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    def esc(text: str) -> str:
        return (
            str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("<", r"$<$")
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}",
        r"\toprule",
        " & ".join(esc(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_tables(results: dict) -> dict[str, tuple[list[str], list[list[str]], str]]:
    models = [m for m in results["models"] if results["models"][m]["summary"]["n_seeds_ok"] > 0]
    summ = {m: results["models"][m]["summary"] for m in models}
    lat = {m: results["models"][m].get("latency") for m in models}
    seeds_note = "mean ± std [95% t-CI] over seeds; '-' std when one seed"
    tables = {}
    tables["main_results"] = (
        ["model", "seeds", "accuracy", "weighted F1", "ECE", "CPU latency (ms)"],
        [
            [
                DISPLAY_NAMES[m],
                str(summ[m]["n_seeds_ok"]),
                fmt_stat(summ[m]["accuracy"]),
                fmt_stat(summ[m]["f1_weighted"]),
                fmt_stat(summ[m]["ece"]),
                f"{lat[m]['mean_ms']:.2f}" if lat[m] else "-",
            ]
            for m in models
        ],
        f"Held-out test set, 3 classes; {seeds_note}. Latency: mean single-prompt CPU inference.",
    )
    tables["per_class_f1"] = (
        ["model"] + [f"F1 {label}" for label in LABELS],
        [[DISPLAY_NAMES[m]] + [fmt_stat(summ[m]["per_class_f1"][label]) for label in LABELS] for m in models],
        f"Per-class F1 on the held-out test set; {seeds_note}.",
    )
    tables["binary_detection"] = (
        ["model", "seeds", "SAFE-vs-attack accuracy", "attack F1"],
        [
            [DISPLAY_NAMES[m], str(summ[m]["n_seeds_ok"]), fmt_stat(summ[m]["binary_accuracy"]), fmt_stat(summ[m]["binary_f1"])]
            for m in models
        ],
        "Binary detection (attack = SUSPICIOUS or MALICIOUS); the only fair comparison for the 2-class baseline.",
    )
    for key, name, caption in (
        ("adversarial_categories", "robustness_by_category",
         "Detection accuracy per category of the hand-written adversarial suite (src/robustness.py)."),
        ("dataset_categories", "test_set_category_detection",
         "Detection accuracy per dataset category on the held-out test set."),
    ):
        cats = sorted({c for m in models for c in summ[m][key]})
        rows = []
        for c in cats:
            n = next(summ[m][key][c]["n"] for m in models if c in summ[m][key])
            rows.append([c, str(n)] + [fmt_stat(summ[m][key].get(c, {}).get("detection_accuracy"), 3) for m in models])
        tables[name] = (["category", "n"] + [DISPLAY_NAMES[m] for m in models], rows, f"{caption} {seeds_note}.")
    return tables


def write_tables(results: dict) -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (columns, rows, caption) in build_tables(results).items():
        md = TABLES_DIR / f"{name}.md"
        md.write_text(f"{caption}\n\n" + to_markdown(columns, rows), encoding="utf-8")
        tex = TABLES_DIR / f"{name}.tex"
        tex.write_text(to_latex(columns, rows, caption, f"tab:{name.replace('_', '-')}"), encoding="utf-8")
        written += [md, tex]
    return written


# --------------------------------------------------------------------------- main


def _environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import sklearn

        env["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import torch
        import transformers

        env.update({"torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda_for_training": torch.cuda.is_available()})
    except Exception as exc:
        env["torch"] = f"unavailable ({type(exc).__name__})"
    return env


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=["classical", "bert", "guard"], default=["classical", "bert", "guard"])
    parser.add_argument("--seeds", type=int, default=5, help="Seeds for classical and bert (default: 5).")
    parser.add_argument("--guard-seeds", type=int, default=1, help="Seeds for guard; ModernBERT is expensive (default: 1).")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke run: cap transformer train/test rows.")
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--protectai", action="store_true", help=f"Also evaluate {PROTECTAI_ID}.")
    args = parser.parse_args(argv)

    _, test_df, provenance = get_splits_with_provenance()
    categories = _category_frame(test_df)
    latency_texts = test_df["text"].sample(n=min(args.latency_samples, len(test_df)), random_state=0).tolist()

    seeds = {
        "classical": _seed_list(args.base_seed, args.seeds),
        "bert": _seed_list(args.base_seed, args.seeds),
        "guard": _seed_list(args.base_seed, args.guard_seeds),
    }
    model_results: dict[str, dict] = {}
    for model_type in args.models:
        if model_type == "classical":
            result = run_classical(seeds["classical"], test_df, categories, latency_texts)
        else:
            result = run_transformer(model_type, seeds[model_type], test_df, categories, latency_texts, args)
        result["seeds_requested"] = seeds[model_type]
        result["summary"] = aggregate(result)
        model_results[model_type] = result
    if args.protectai:
        result = run_protectai(test_df, categories, latency_texts)
        result["seeds_requested"] = []
        result["summary"] = aggregate(result)
        model_results["protectai"] = result

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": provenance,
        "config": {
            **vars(args),
            "evaluation_device": "cpu",
            "inputs": "test texts are the clean_text-normalized dataset rows, as the API sees them",
            "confidence_interval": "two-sided 95% Student-t over seeds",
            "min_category_rows": MIN_CATEGORY_ROWS,
            "smoke_run": bool(args.max_samples),
        },
        "environment": _environment(),
        "models": model_results,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    written = write_tables(results)

    print("\n" + build_tables_text(results))
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)} and {len(written)} table files in {TABLES_DIR.relative_to(ROOT)}/")
    for model_type, result in model_results.items():
        s = result["summary"]
        if s["n_seeds_failed"]:
            first = next(r for r in result["runs"] if r["status"] != "ok")
            print(f"WARNING: {model_type}: {s['n_seeds_failed']} run(s) failed: {first['reason'][:200]}")
    return results


def build_tables_text(results: dict) -> str:
    columns, rows, caption = build_tables(results)["main_results"]
    return caption + "\n" + to_markdown(columns, rows)


if __name__ == "__main__":
    main()
