"""Why do the models fail on external data? Diagnostics for src.external_eval.

    python -m src.external_diagnostics                  # classical (+ bert/guard if they load)
    python -m src.external_diagnostics --skip-multi-source

Per external test set (xTRam1, deepset; Qualifire is gated and not used here):
1. Label sanity: binary accuracy under the documented mapping (1 = injection)
   and under inverted labels. If inverted were far higher, the mapping is wrong.
   A fixed random sample of 30 rows (seed 2026) is printed and saved for manual
   inspection; the manual verdict is recorded in docs/external_label_check.md.
2. Error analysis: the 20 most confident false positives and false negatives,
   plus simple feature summaries (length, instruction-style phrasing, language).
3. deepset language split: a stopword heuristic separates English from German
   rows and metrics are reported per subset. PromptGuard does not claim
   multilingual support; the training data is English.
4. MULTI-SOURCE TRAINING (clearly separate, classical only): leave-one-dataset-out.
   Train on neuralchemy train + the OTHER external dataset's TRAIN split, select
   on the neuralchemy validation split, test on the held-out dataset's TEST
   split. External test splits are never used for training or tuning, and these
   results never replace the single-source headline.

Output: artifacts/external_diagnostics.json and artifacts/tables/external_diagnostics.md.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.external_eval import DATASETS, ExternalDataset, load_external, normalize, remove_training_overlap, training_texts
from src.metrics import LABELS, summarize_binary
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "external_diagnostics.json"
TABLE_PATH = ROOT / "artifacts" / "tables" / "external_diagnostics.md"
SAMPLE_SEED = 2026
SAMPLE_N = 30
TOP_K = 20
USED = [ds for ds in DATASETS if not ds.gated]

EN_STOPWORDS = {"the", "and", "is", "are", "to", "of", "what", "how", "you", "your", "a", "in", "for", "that", "with",
                "this", "it", "do", "can", "i", "my", "me", "be", "on", "why", "will"}
DE_STOPWORDS = {"der", "die", "das", "und", "ist", "nicht", "ich", "wie", "was", "ein", "eine", "zu", "mit", "für",
                "auf", "sie", "es", "den", "dem", "von", "wir", "sind", "warum", "welche", "gibt", "im", "auch"}
INSTRUCTION_STYLE = re.compile(
    r"\b(your task|you are given|you are|answer the following|options:|premise|hypothesis|step-by-step|"
    r"given the|in this task|question:)",
    flags=re.IGNORECASE,
)


def language_of(text: str) -> str:
    """'en', 'de' or 'unknown' by stopword counts (a heuristic, not a language model)."""
    words = re.findall(r"[a-zäöüß]+", text.lower())
    en = sum(w in EN_STOPWORDS for w in words)
    de = sum(w in DE_STOPWORDS for w in words)
    if en == de:
        return "unknown"
    return "en" if en > de else "de"


def _external_train(ds: ExternalDataset) -> pd.DataFrame:
    """The dataset's TRAIN split mapped to SAFE/MALICIOUS (for multi-source training only)."""
    train_ds = ExternalDataset(**{**ds.__dict__, "split": "train"})
    return load_external(train_ds)


def _probs(predict, texts: list[str]) -> np.ndarray:
    return np.concatenate([predict(texts[i : i + 64]) for i in range(0, len(texts), 64)])


def _features(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    return {
        "n": int(len(frame)),
        "median_chars": float(frame["text"].str.len().median()),
        "instruction_style_share": float(frame["text"].map(lambda t: bool(INSTRUCTION_STYLE.search(t))).mean()),
        "language_share": {k: float(v) for k, v in frame["text"].map(language_of).value_counts(normalize=True).items()},
    }


def diagnose(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    y_true = frame["label"].tolist()
    p_safe = probs[:, LABELS.index("SAFE")]
    pred_attack = probs.argmax(axis=1) != LABELS.index("SAFE")
    true_attack = frame["label"].eq("MALICIOUS").to_numpy()
    inverted = ["SAFE" if label == "MALICIOUS" else "MALICIOUS" for label in y_true]
    out = {
        "metrics": summarize_binary(y_true, probs),
        "inverted_label_accuracy": summarize_binary(inverted, probs)["accuracy"],
    }
    fp = frame.assign(p_attack=1 - p_safe)[~true_attack & pred_attack].sort_values("p_attack", ascending=False)
    fn = frame.assign(p_safe=p_safe)[true_attack & ~pred_attack].sort_values("p_safe", ascending=False)
    tn = frame[~true_attack & ~pred_attack]
    out["top_false_positives"] = [{"p_attack": round(float(r.p_attack), 4), "text": r.text[:240]} for r in fp.head(TOP_K).itertuples()]
    out["top_false_negatives"] = [{"p_safe": round(float(r.p_safe), 4), "text": r.text[:240]} for r in fn.head(TOP_K).itertuples()]
    out["features"] = {"false_positives": _features(fp), "true_negatives": _features(tn), "false_negatives": _features(fn)}
    return out


def multi_source(held_out: ExternalDataset, other: ExternalDataset, test_frame: pd.DataFrame) -> dict:
    """Leave-one-dataset-out with the classical model. Labelled MULTI-SOURCE everywhere."""
    from src.train_classical import _ordered_proba, fit_classical

    extra = _external_train(other)
    held_exact, held_norm = set(test_frame["text"]), set(test_frame["text"].map(normalize))
    overlap = extra["text"].isin(held_exact) | extra["text"].map(normalize).isin(held_norm)
    extra = extra[~overlap]
    extra = extra.assign(text=extra["text"].map(clean_text))[["text", "label"]]
    fitted = fit_classical(random_state=42, extra_train=extra)
    probs = _ordered_proba(fitted["pipeline"], [clean_text(t) for t in test_frame["text"]])
    return {
        "setting": "MULTI-SOURCE TRAINING (not the single-source headline)",
        "train": f"neuralchemy train + {other.name} train split",
        "extra_train_rows": int(len(extra)),
        "extra_rows_dropped_as_overlapping_held_out_test": int(overlap.sum()),
        "selection": "neuralchemy validation split only (no external test data used for tuning)",
        "selected_model": fitted["best_name"],
        "held_out_test": held_out.name,
        "metrics": summarize_binary(test_frame["label"].tolist(), probs),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-multi-source", action="store_true")
    args = parser.parse_args(argv)

    from src.external_eval import load_models

    exact, normalized, provenance = training_texts()
    models, skipped = load_models(include_protectai=False)
    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_dataset": provenance,
        "models_skipped": skipped,
        "language_heuristic": "stopword counts (English vs German); PromptGuard does not claim multilingual support",
        "datasets": {},
    }
    frames = {}
    for ds in USED:
        frame, dedup = remove_training_overlap(load_external(ds), exact, normalized)
        frames[ds.name] = frame
        entry = {"revision": ds.revision, "license": ds.license, "evaluated_rows": int(len(frame)), **dedup}
        sample = frame.sample(min(SAMPLE_N, len(frame)), random_state=SAMPLE_SEED)
        entry["manual_check_sample"] = {
            "seed": SAMPLE_SEED,
            "rows": [{"label": str(r.raw_label), "text": r.text[:200]} for r in sample.itertuples()],
            "verdict": "see docs/external_label_check.md",
        }
        texts = [clean_text(t) for t in frame["text"]]
        entry["models"] = {}
        for name, predict in models.items():
            probs = _probs(predict, texts)
            diag = diagnose(frame, probs)
            if ds.name.startswith("deepset/"):
                lang = frame["text"].map(language_of)
                diag["by_language"] = {
                    code: {**summarize_binary(frame.loc[lang == code, "label"].tolist(), probs[(lang == code).to_numpy()]),
                           "rows": int((lang == code).sum())}
                    for code in ("en", "de")
                    if (lang == code).sum() > 0
                }
            entry["models"][name] = diag
            m = diag["metrics"]
            print(
                f"{ds.name} / {name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} fpr={m['false_positive_rate']:.4f} "
                f"| inverted-label acc={diag['inverted_label_accuracy']:.4f}"
            )
            if "by_language" in diag:
                for code, lm in diag["by_language"].items():
                    print(f"    {code}: rows={lm['rows']} acc={lm['accuracy']:.4f} f1={lm['f1']:.4f} fpr={lm['false_positive_rate']}")
            f = diag["features"]
            print(f"    features FP={f['false_positives']}  TN={f['true_negatives']}")
        results["datasets"][ds.name] = entry

    if not args.skip_multi_source and "classical" in models:
        results["multi_source_training"] = []
        for held in USED:
            other = next(ds for ds in USED if ds.name != held.name)
            print(f"\nMULTI-SOURCE TRAINING: held out {held.name}, extra train = {other.name} train split")
            res = multi_source(held, other, frames[held.name])
            results["multi_source_training"].append(res)
            m = res["metrics"]
            print(f"    selected {res['selected_model']}; acc={m['accuracy']:.4f} f1={m['f1']:.4f} fpr={m['false_positive_rate']:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"External diagnostics. Training data: {provenance['name']} @ {provenance['revision'][:7]}.",
        "",
        "| dataset | model | setting | rows | accuracy | F1 | FPR | inverted-label acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in results["datasets"].items():
        for model, diag in entry["models"].items():
            m = diag["metrics"]
            lines.append(f"| {name} | {model} | single-source | {entry['evaluated_rows']} | {m['accuracy']:.4f} | "
                         f"{m['f1']:.4f} | {m['false_positive_rate']:.4f} | {diag['inverted_label_accuracy']:.4f} |")
            for code, lm in diag.get("by_language", {}).items():
                lines.append(f"| {name} [{code} subset] | {model} | single-source | {lm['rows']} | {lm['accuracy']:.4f} | "
                             f"{lm['f1']:.4f} | {lm['false_positive_rate']} | - |")
    for res in results.get("multi_source_training", []):
        m = res["metrics"]
        lines.append(f"| {res['held_out_test']} | classical | MULTI-SOURCE TRAINING (+{res['train'].split(' + ')[1]}) | "
                     f"{m['n_samples']} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['false_positive_rate']:.4f} | - |")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)} and {TABLE_PATH.relative_to(ROOT)}")
    return results


if __name__ == "__main__":
    main()
