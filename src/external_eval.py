"""Evaluate the trained models, without retraining, on external test sets.

    python -m src.external_eval                    # classical, bert, guard (those that load)
    python -m src.external_eval --protectai        # plus the ProtectAI baseline

Datasets (pinned revisions):
* xTRam1/safe-guard-prompt-injection, test split - no license stated on the card
* deepset/prompt-injections, test split - Apache-2.0 (mixed English and German text)
* qualifire/prompt-injections-benchmark, test split - CC-BY-NC-4.0 and gated
  (accept the terms on the Hub and set HF_TOKEN); skipped with a message otherwise

Labels are mapped to SAFE vs MALICIOUS. The xTRam1 and deepset cards do not
document the label meaning, so each run re-checks it: the share of texts with
injection keywords must be higher for the label mapped to MALICIOUS, and
sample texts per label are stored for audit.

Before scoring, texts that also occur in the neuralchemy training data (all
rows of data/train.csv, plus the classical model's train-only extras) are
removed - exact matches and normalized matches (trimmed, lowercased,
whitespace collapsed) - and the counts are logged per dataset.

Reports accuracy, precision, recall, F1 (MALICIOUS), false-positive rate and
ECE per model per dataset (fixed threshold: the argmax label the API returns),
plus, per model and per dataset (and the English subset of deepset):
* the 3-class prediction distribution on true-SAFE and true-MALICIOUS rows;
* binary AUROC of P(not-SAFE) with a bootstrap 95% CI (threshold-free);
* metrics at the threshold that maximizes accuracy on the neuralchemy
  VALIDATION split - chosen once per model, never on the external data.
Output: artifacts/external_eval.json and artifacts/tables/external_eval.{md,tex},
external_eval_thresholds.md.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import probs_io
from src.metrics import LABELS, summarize_binary
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "external_eval.json"
TABLES_DIR = ROOT / "artifacts" / "tables"

INJECTION_KEYWORDS = re.compile(
    r"\b(ignore|ignoriere|ignorieren|vergiss|vergessen|forget|disregard|override|jailbreak|pretend|"
    r"system prompt|instructions|anweisungen|reveal|bypass|dan)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExternalDataset:
    name: str
    revision: str
    split: str
    license: str
    text_column: str = "text"
    label_column: str = "label"
    malicious_values: tuple = (1,)
    benign_values: tuple = (0,)
    notes: str = ""
    gated: bool = False
    extra: dict = field(default_factory=dict)


DATASETS = [
    ExternalDataset(
        name="xTRam1/safe-guard-prompt-injection",
        revision="a3a877d608f37b7d20d9945671902df895ecdb46",
        split="test",
        license="none stated on the dataset card",
        notes="Label meaning undocumented on the card; checked from samples and keyword rates (1 = injection).",
    ),
    ExternalDataset(
        name="deepset/prompt-injections",
        revision="4f61ecb038e9c3fb77e21034b22511b523772cdd",
        split="test",
        license="Apache-2.0",
        notes="Mixed English and German text; label meaning undocumented on the card; checked (1 = injection).",
    ),
    ExternalDataset(
        name="qualifire/prompt-injections-benchmark",
        revision="9ef1aa46a7e5eedb096be0481be8011ede1e72e8",
        split="test",
        license="CC-BY-NC-4.0 (non-commercial)",
        gated=True,
        notes="Gated on the Hub: accept the terms and set HF_TOKEN. Label column/values are verified at load time.",
        malicious_values=("jailbreak", "injection", "prompt_injection", "malicious", 1),
        benign_values=("benign", "safe", 0),
    ),
]


def normalize(text: str) -> str:
    """Trim, lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def training_texts() -> tuple[set[str], set[str], dict]:
    """Exact and normalized texts of everything the models were trained on."""
    from src.load_data import SPLIT_PATHS, dataset_provenance
    from src.train_classical import EVERYDAY_SAFE_EXAMPLES

    train = pd.read_csv(SPLIT_PATHS["train"], keep_default_na=False)
    texts = list(train["text"].astype(str)) + list(EVERYDAY_SAFE_EXAMPLES)
    return set(texts), {normalize(t) for t in texts}, dataset_provenance()


def load_external(ds: ExternalDataset) -> pd.DataFrame:
    from datasets import load_dataset

    frame = load_dataset(ds.name, split=ds.split, revision=ds.revision).to_pandas()
    if ds.text_column not in frame or ds.label_column not in frame:
        raise ValueError(f"Expected columns {ds.text_column!r}/{ds.label_column!r}, found {list(frame.columns)}")
    raw = frame[ds.label_column]
    normalized = raw.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    mapping = {v: "MALICIOUS" for v in ds.malicious_values} | {v: "SAFE" for v in ds.benign_values}
    unknown = sorted({str(v) for v in normalized.unique() if v not in mapping})
    if unknown:
        raise ValueError(f"Unrecognized label values {unknown}; inspect samples and extend the mapping first.")
    return pd.DataFrame({"text": frame[ds.text_column].astype(str), "raw_label": raw, "label": normalized.map(mapping)})


def label_check(frame: pd.DataFrame) -> dict:
    """Keyword-rate sanity check of the label mapping, plus samples for audit."""
    rates = {
        label: float(frame.loc[frame["label"] == label, "text"].map(lambda t: bool(INJECTION_KEYWORDS.search(t))).mean())
        for label in ("SAFE", "MALICIOUS")
    }
    samples = {
        label: frame.loc[frame["label"] == label, "text"].head(3).map(lambda t: t[:160]).tolist()
        for label in ("SAFE", "MALICIOUS")
    }
    return {
        "injection_keyword_rate": rates,
        "passed": rates["MALICIOUS"] > rates["SAFE"],
        "samples": samples,
        "raw_label_counts": {str(k): int(v) for k, v in frame["raw_label"].value_counts().items()},
    }


def remove_training_overlap(frame: pd.DataFrame, exact: set[str], normalized: set[str]) -> tuple[pd.DataFrame, dict]:
    is_exact = frame["text"].isin(exact)
    is_norm = ~is_exact & frame["text"].map(normalize).isin(normalized)
    kept = frame[~(is_exact | is_norm)].reset_index(drop=True)
    return kept, {"exact_duplicates_removed": int(is_exact.sum()), "normalized_duplicates_removed": int(is_norm.sum())}


def load_models(include_protectai: bool, baseline_names: tuple[str, ...] = (),
                include_trained: bool = True) -> tuple[dict, dict]:
    """name -> callable(texts) -> (n, 3) probabilities; plus skip reasons.

    include_trained=False loads only ProtectAI / the named baselines (used to add a model to existing results).
    """
    from src.predict import PromptAnalyzer

    models, skipped = {}, {}
    for model_type in ("classical", "bert", "guard") if include_trained else ():
        try:
            analyzer = PromptAnalyzer(model_type=model_type, device="cpu")
            models[model_type] = analyzer.predict_proba
        except Exception as exc:
            skipped[model_type] = f"{type(exc).__name__}: {exc}"
    if include_protectai:
        try:
            from src.paper_results import PROTECTAI_ID, TransformerPredictor

            models["protectai"] = TransformerPredictor(PROTECTAI_ID, max_length=512).predict
        except Exception as exc:
            skipped["protectai"] = f"{type(exc).__name__}: {exc}"
    from src.baselines import FACTORIES

    for name in baseline_names:
        try:
            models[name] = FACTORIES[name]().predict
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return models, skipped


AUROC_BOOTSTRAP = 1000
AUROC_SEED = 2026


def attack_score(probs: np.ndarray) -> np.ndarray:
    """P(not-SAFE) = 1 - P(SAFE): the binary score thresholded below."""
    return 1.0 - np.asarray(probs, dtype=float)[:, LABELS.index("SAFE")]


def prediction_distribution(y_true: list[str], probs: np.ndarray) -> dict:
    """3-class argmax counts on true-SAFE and on true-MALICIOUS rows."""
    predicted = np.asarray(LABELS)[np.asarray(probs).argmax(axis=1)]
    y = np.asarray(y_true)
    return {
        truth: {label: int(((y == truth) & (predicted == label)).sum()) for label in LABELS}
        for truth in ("SAFE", "MALICIOUS")
    }


def auroc_with_ci(y_true: list[str], score: np.ndarray, n_boot: int = AUROC_BOOTSTRAP, seed: int = AUROC_SEED) -> dict:
    """Binary AUROC (MALICIOUS positive) with a percentile bootstrap 95% CI over rows."""
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(y_true) == "MALICIOUS").astype(int)
    score = np.asarray(score, dtype=float)
    if y.min() == y.max():
        return {"auroc": None, "ci95": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], score[idx]))
    return {
        "auroc": float(roc_auc_score(y, score)),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_boot": len(boots),
    }


def best_validation_threshold(y_true: list[str], score: np.ndarray) -> dict:
    """Threshold on P(not-SAFE) that maximizes binary accuracy on the given
    (validation) rows; ties go to the threshold closest to 0.5. Predict
    MALICIOUS when score >= threshold."""
    y = np.asarray(y_true) == "MALICIOUS"
    score = np.asarray(score, dtype=float)
    values = np.unique(score)
    candidates = np.unique(np.concatenate([[0.5], (values[:-1] + values[1:]) / 2, [values[0], values[-1] + 1e-9]]))
    accuracy = np.array([((score >= t) == y).mean() for t in candidates])
    best = np.flatnonzero(accuracy == accuracy.max())
    t = float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])
    return {"threshold": t, "validation_accuracy": float(accuracy.max()), "validation_rows": int(len(y))}


def metrics_at_threshold(y_true: list[str], score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y_true) == "MALICIOUS"
    pred = np.asarray(score, dtype=float) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "n_samples": int(len(y)),
    }


def validation_thresholds(models: dict) -> dict:
    """Per model: accuracy-maximizing threshold on the neuralchemy VALIDATION
    split (original Hub rows, the binary headline view). Never the external data."""
    from src.preprocess import load_split

    val = load_split("validation")
    val = val[val["derivation"] == "original"]
    y = ["SAFE" if label == "SAFE" else "MALICIOUS" for label in val["label"]]
    out = {}
    for name, predict in models.items():
        print(f"[{name}] choosing a threshold on the neuralchemy validation split ({len(val)} original rows) ...")
        out[name] = best_validation_threshold(y, attack_score(_batched(predict, val["text"].tolist())))
    return out


def score_external(y_true: list[str], probs: np.ndarray, threshold: dict | None) -> dict:
    """Fixed-threshold (argmax) metrics plus the threshold-free and validation-tuned views."""
    score = attack_score(probs)
    out = summarize_binary(y_true, probs)
    out["prediction_distribution"] = prediction_distribution(y_true, probs)
    out["auroc"] = auroc_with_ci(y_true, score)
    if threshold is not None:
        out["validation_tuned"] = metrics_at_threshold(y_true, score, threshold["threshold"])
    return out


def weights_source(predict) -> dict:
    """Where a loaded model's weights came from, recorded at load time.

    *predict* is a bound method of a PromptAnalyzer or TransformerPredictor.
    Returns {"source": local path (relative to the repo when inside it) or Hub repo id,
    "loaded_from": "local" | "hub", "hub_revision": commit hash for Hub loads, else None}.
    """
    owner = getattr(predict, "__self__", None)
    source = getattr(owner, "source", None)
    if source is None:
        return {"source": None, "loaded_from": None, "hub_revision": None}
    model = getattr(owner, "model", None)
    model_type = getattr(owner, "model_type", None)
    if model_type in ("bert", "guard"):
        model = getattr(owner, f"{model_type}_model", None)
    if model is None and model_type is None:  # rule-based baselines (src.baselines): no weights at all
        return {"source": str(source), "loaded_from": "code", "hub_revision": None}
    path = Path(str(source))
    if path.exists():
        resolved = path.resolve()
        shown = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)
        return {"source": shown, "loaded_from": "local", "hub_revision": None}
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    return {"source": str(source), "loaded_from": "hub", "hub_revision": revision}


def _batched(predict, texts: list[str], size: int = 64) -> np.ndarray:
    return np.concatenate([predict(texts[i : i + size]) for i in range(0, len(texts), size)])


def _skip_reason(ds: ExternalDataset, exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()
    if ds.gated or "gated" in lowered or "401" in lowered or "403" in lowered:
        return (
            f"not downloadable without access: {ds.name} is gated ({ds.license}). Accept its terms on "
            f"https://huggingface.co/datasets/{ds.name} and set HF_TOKEN, then rerun. ({message[:200]})"
        )
    if any(m in lowered for m in ("connection", "resolve", "offline", "timed out", "max retries")):
        return f"network unavailable: {message[:200]}"
    return message[:300]


def to_markdown(columns: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def to_latex(columns: list[str], rows: list[list[str]], caption: str) -> str:
    def esc(text: str) -> str:
        return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")

    body = [" & ".join(esc(v) for v in r) + r" \\" for r in rows]
    return "\n".join(
        [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{ll" + "r" * (len(columns) - 2) + "}",
         r"\toprule", " & ".join(esc(c) for c in columns) + r" \\", r"\midrule", *body, r"\bottomrule",
         r"\end{tabular}", rf"\caption{{{esc(caption)}}}", r"\label{tab:external-eval}", r"\end{table}"]
    ) + "\n"


def _dist(counts: dict) -> str:
    return " / ".join(str(counts[label]) for label in LABELS)


def threshold_table(datasets_out: list[dict]) -> tuple[list[str], list[list[str]]]:
    columns = ["dataset", "model", "n", "true SAFE -> S / SU / M", "true MAL -> S / SU / M", "AUROC [95% CI]",
               "fixed acc", "tuned threshold", "tuned acc", "tuned FPR"]
    rows = []
    for entry in datasets_out:
        if entry["status"] != "ok":
            continue
        groups = [(entry["name"], entry["models"])] + [
            (f"{entry['name']} [{code} subset]", sub["models"]) for code, sub in entry.get("subsets", {}).items()
        ]
        for label, per_model in groups:
            for name, m in per_model.items():
                auc = m["auroc"]
                auc_text = "-" if auc["auroc"] is None else f"{auc['auroc']:.4f} [{auc['ci95'][0]:.4f}, {auc['ci95'][1]:.4f}]"
                tuned = m.get("validation_tuned")
                fpr = None if tuned is None else tuned["false_positive_rate"]
                rows.append([
                    label, name, str(m["n_samples"]), _dist(m["prediction_distribution"]["SAFE"]),
                    _dist(m["prediction_distribution"]["MALICIOUS"]), auc_text, f"{m['accuracy']:.4f}",
                    "-" if tuned is None else f"{tuned['threshold']:.4f}",
                    "-" if tuned is None else f"{tuned['accuracy']:.4f}",
                    "-" if fpr is None else f"{fpr:.4f}",
                ])
    return columns, rows


def merge_results(existing: dict, added: dict) -> dict:
    """Add the models of *added* to *existing* (same datasets, same rows); existing models are kept."""
    by_name = {d["name"]: d for d in added["datasets"]}
    for entry in existing["datasets"]:
        new = by_name.get(entry["name"])
        if entry.get("status") != "ok" or not new or new.get("status") != "ok":
            continue
        if new["evaluated_rows"] != entry["evaluated_rows"]:
            raise ValueError(f"{entry['name']}: {new['evaluated_rows']} rows now vs {entry['evaluated_rows']} before")
        entry["models"].update(new["models"])
        for code, subset in new.get("subsets", {}).items():
            entry.setdefault("subsets", {}).setdefault(code, subset)["models"].update(subset["models"])
    for key in ("models_skipped", "model_weights", "validation_thresholds"):
        existing.setdefault(key, {}).update(added.get(key, {}))
    existing["generated_at"] = added["generated_at"]
    existing.setdefault("added_later", []).append({"models": sorted(added["model_weights"]), "at": added["generated_at"]})
    return existing


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protectai", action="store_true", help="Also evaluate protectai/deberta-v3-base-prompt-injection-v2.")
    parser.add_argument("--baselines", nargs="*", default=[], choices=["regex", "keyword", "llama_pg2"],
                        help="Untrained baselines from src.baselines (llama_pg2 needs HF_TOKEN with the licence accepted).")
    parser.add_argument("--add-only", action="store_true",
                        help="Evaluate only --protectai / --baselines and merge them into the existing "
                             "artifacts/external_eval.json instead of re-running every model.")
    args = parser.parse_args(argv)

    exact, normalized, provenance = training_texts()
    models, model_skips = load_models(args.protectai, tuple(args.baselines), include_trained=not args.add_only)
    if args.add_only and not models:
        raise SystemExit(f"--add-only: no model could be loaded ({model_skips})")
    for name, reason in model_skips.items():
        print(f"[model {name}] skipped: {reason[:200]}")
    model_weights = {name: weights_source(predict) for name, predict in models.items()}
    for name, info in model_weights.items():
        revision = f" @ {info['hub_revision']}" if info["hub_revision"] else ""
        print(f"[model {name}] weights: {info['source']} ({info['loaded_from']}{revision})")
    thresholds = validation_thresholds(models)

    datasets_out = []
    for ds in DATASETS:
        entry = {"name": ds.name, "revision": ds.revision, "split": ds.split, "license": ds.license, "notes": ds.notes}
        print(f"\n== {ds.name} ({ds.split} @ {ds.revision[:7]}, license: {ds.license})")
        try:
            frame = load_external(ds)
        except Exception as exc:
            entry.update({"status": "skipped", "reason": _skip_reason(ds, exc)})
            print(f"   skipped: {entry['reason']}")
            datasets_out.append(entry)
            continue
        check = label_check(frame)
        entry["split_size"] = int(len(frame))
        entry["label_mapping"] = {"MALICIOUS": list(map(str, ds.malicious_values)), "SAFE": list(map(str, ds.benign_values))}
        entry["label_check"] = check
        if not check["passed"]:
            entry.update({"status": "failed", "reason": "label check failed: MALICIOUS keyword rate not above SAFE"})
            print(f"   FAILED label check: {check['injection_keyword_rate']}")
            datasets_out.append(entry)
            continue
        kept, dedup = remove_training_overlap(frame, exact, normalized)
        entry.update(dedup)
        entry["evaluated_rows"] = int(len(kept))
        entry["evaluated_label_counts"] = {str(k): int(v) for k, v in kept["label"].value_counts().items()}
        print(
            f"   rows={len(frame)} label check keyword rates={check['injection_keyword_rate']} -> "
            f"removed exact={dedup['exact_duplicates_removed']} normalized={dedup['normalized_duplicates_removed']}, "
            f"evaluating {len(kept)}"
        )
        texts = [clean_text(t) for t in kept["text"]]  # same normalization as the API
        subsets = {}
        if ds.name.startswith("deepset/"):
            from src.external_diagnostics import language_of

            subsets["en"] = (kept["text"].map(language_of) == "en").to_numpy()
        entry["models"] = {}
        entry["subsets"] = {code: {"rows": int(mask.sum()), "selection": "stopword heuristic (src.external_diagnostics.language_of)",
                                   "models": {}} for code, mask in subsets.items()}
        y_true = kept["label"].tolist()
        for name, predict in models.items():
            probs = _batched(predict, texts)
            probs_io.save_probs(
                f"external_{probs_io.slug(ds.name)}_{name}", probs, y_true, texts,
                **{f"subset_{code}": mask for code, mask in subsets.items()},
            )
            metrics = score_external(y_true, probs, thresholds.get(name))
            entry["models"][name] = metrics
            for code, mask in subsets.items():
                entry["subsets"][code]["models"][name] = score_external(
                    [y for y, keep in zip(y_true, mask) if keep], probs[mask], thresholds.get(name)
                )
            tuned = metrics.get("validation_tuned", {})
            print(
                f"   {name:<10} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"fpr={metrics['false_positive_rate']:.4f} ece={metrics['ece']:.4f} auroc={metrics['auroc']['auroc']:.4f} "
                f"| val-threshold {tuned.get('threshold', float('nan')):.4f}: acc={tuned.get('accuracy', float('nan')):.4f}"
            )
        entry["status"] = "ok"
        datasets_out.append(entry)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_dataset": provenance,
        "dedup_reference": "all rows of data/train.csv plus the classical model's train-only extras",
        "models_skipped": model_skips,
        "model_weights": model_weights,
        "views": {
            "fixed_threshold": "argmax of the 3-class output (the label the API returns); no tuning",
            "auroc": "threshold-free; score = P(not-SAFE) = 1 - P(SAFE); percentile bootstrap 95% CI "
                     f"({AUROC_BOOTSTRAP} resamples of rows, seed {AUROC_SEED})",
            "validation_tuned": "threshold on P(not-SAFE) that maximizes binary accuracy on the neuralchemy "
                                "VALIDATION split (original rows); chosen once per model, never on external data",
        },
        "validation_thresholds": thresholds,
        "datasets": datasets_out,
    }
    if args.add_only:
        results = merge_results(json.loads(OUTPUT_PATH.read_text(encoding="utf-8")), results)
        datasets_out = results["datasets"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    columns = ["dataset", "model", "n", "dups removed", "accuracy", "F1 (MALICIOUS)", "FPR", "ECE"]
    rows = []
    for entry in datasets_out:
        if entry["status"] != "ok":
            rows.append([entry["name"], "-", "-", "-", "skipped", "-", "-", "-"])
            continue
        removed = entry["exact_duplicates_removed"] + entry["normalized_duplicates_removed"]
        for name, m in entry["models"].items():
            rows.append([entry["name"], name, str(entry["evaluated_rows"]), str(removed), f"{m['accuracy']:.4f}",
                         f"{m['f1']:.4f}", f"{m['false_positive_rate']:.4f}", f"{m['ece']:.4f}"])
    caption = (
        f"External test sets, no retraining; SAFE vs MALICIOUS. Training data: {provenance['name']} "
        f"@ {provenance['revision'][:7]} ({provenance['train_config']}). Texts also in the training data removed first."
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "external_eval.md").write_text(f"{caption}\n\n" + to_markdown(columns, rows), encoding="utf-8")
    (TABLES_DIR / "external_eval.tex").write_text(to_latex(columns, rows, caption), encoding="utf-8")

    t_columns, t_rows = threshold_table(datasets_out)
    t_caption = (
        "External test sets: prediction distribution and AUROC. Fixed = argmax label (no tuning); AUROC is "
        "threshold-free on P(not-SAFE) with a bootstrap 95% CI; tuned = threshold chosen on the neuralchemy "
        "VALIDATION split only (never on external data)."
    )
    (TABLES_DIR / "external_eval_thresholds.md").write_text(f"{t_caption}\n\n" + to_markdown(t_columns, t_rows), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)} and artifacts/tables/external_eval.md|.tex, external_eval_thresholds.md")
    return results


if __name__ == "__main__":
    main()
