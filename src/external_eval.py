"""Evaluate the trained models, without retraining, on external test sets.

    python -m src.external_eval                    # classical, bert, guard (those that load)
    python -m src.external_eval --protectai        # plus the ProtectAI baseline

Datasets (pinned revisions):
* xTRam1/safe-guard-prompt-injection, test split - no license stated on the card
* deepset/prompt-injections, test split - Apache-2.0 (mostly German text)
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
ECE per model per dataset. Output: artifacts/external_eval.json and
artifacts/tables/external_eval.{md,tex}.
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

from src.metrics import summarize_binary
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
        notes="Mostly German text; label meaning undocumented on the card; checked (1 = injection).",
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


def load_models(include_protectai: bool) -> tuple[dict, dict]:
    """name -> callable(texts) -> (n, 3) probabilities; plus skip reasons."""
    from src.predict import PromptAnalyzer

    models, skipped = {}, {}
    for model_type in ("classical", "bert", "guard"):
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
    return models, skipped


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


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protectai", action="store_true", help="Also evaluate protectai/deberta-v3-base-prompt-injection-v2.")
    args = parser.parse_args(argv)

    exact, normalized, provenance = training_texts()
    models, model_skips = load_models(args.protectai)
    for name, reason in model_skips.items():
        print(f"[model {name}] skipped: {reason[:200]}")

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
        entry["models"] = {}
        for name, predict in models.items():
            metrics = summarize_binary(kept["label"].tolist(), _batched(predict, texts))
            entry["models"][name] = metrics
            print(
                f"   {name:<10} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"fpr={metrics['false_positive_rate']:.4f} ece={metrics['ece']:.4f}"
            )
        entry["status"] = "ok"
        datasets_out.append(entry)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_dataset": provenance,
        "dedup_reference": "all rows of data/train.csv plus the classical model's train-only extras",
        "models_skipped": model_skips,
        "datasets": datasets_out,
    }
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
    print(f"\nWrote {OUTPUT_PATH.relative_to(ROOT)} and artifacts/tables/external_eval.md|.tex")
    return results


if __name__ == "__main__":
    main()
