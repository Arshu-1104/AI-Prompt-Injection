"""Evaluate the decision pipeline the API runs, not only the raw models.

For model=classical, src.predict.PromptAnalyzer.predict does:
1. classical label and confidence (max class probability);
2. escalation, when enabled and the guard model is loaded: if the classical label
   is SUSPICIOUS and its confidence is below 0.65, the guard label replaces it;
3. rule floor: a chat-template token in the raw text, or content hidden entirely
   in tags, raises SAFE to SUSPICIOUS.

This script measures:
* escalation: a sweep of the threshold t for two trigger rules,
    suspicious_only - the deployed rule (label == SUSPICIOUS and confidence < t)
    any_label       - escalate whenever the classical confidence is below t
  with binary accuracy, precision, recall, F1, FPR and the share escalated on the
  validation split, the test split (original rows, as the headline) and each
  external set. t is chosen per rule on the VALIDATION split only (highest binary
  accuracy, ties -> fewer escalations) and then reported on the other sets.
* latency: per-text CPU time at batch size 1 for classical and guard on a fixed
  sample of 200 test texts; a pipeline's mean latency at t is the mean over that
  sample of classical time + guard time for the texts that escalate at t.
* rule floor: how many rows it changes, and the metrics with and without it.

Inputs: models/classical_model.pkl (the deployed classical model) and the guard
weights in artifacts/paper_runs/guard/seed_42 (the run uploaded to the Hub).
External sets reuse the probability files of src.external_eval.
Outputs: artifacts/pipeline_eval.json, artifacts/tables/escalation.{md,tex},
artifacts/tables/rule_floor.{md,tex}; probabilities in artifacts/probs/.

    python -m src.pipeline_eval
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import probs_io
from src.metrics import LABELS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_PATH = ARTIFACTS / "pipeline_eval.json"
TABLES_DIR = ARTIFACTS / "tables"
GUARD_DIR = ARTIFACTS / "paper_runs" / "guard" / "seed_42"
CLASSICAL_PATH = ROOT / "models" / "classical_model.pkl"
DEPLOYED_THRESHOLD = 0.65  # src.predict.PromptAnalyzer.predict
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.35, 1.0001, 0.05)) + (1.01,)  # 1.01 = escalate everything eligible
RULES = ("suspicious_only", "any_label")
LATENCY_SAMPLES = 200
WARMUP = 3
SAFE = LABELS.index("SAFE")
SUSPICIOUS = LABELS.index("SUSPICIOUS")


# --------------------------------------------------------------------------- decision logic


def escalate_mask(classical_probs: np.ndarray, rule: str, threshold: float) -> np.ndarray:
    """Rows the pipeline sends to the guard model."""
    probs = np.asarray(classical_probs)
    confidence = probs.max(axis=1)
    if rule == "suspicious_only":
        return (probs.argmax(axis=1) == SUSPICIOUS) & (confidence < threshold)
    if rule == "any_label":
        return confidence < threshold
    raise ValueError(f"unknown rule {rule!r}")


def pipeline_labels(classical_probs: np.ndarray, guard_probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Label index per row: guard's where escalated, classical's elsewhere."""
    return np.where(mask, np.asarray(guard_probs).argmax(axis=1), np.asarray(classical_probs).argmax(axis=1))


def apply_rule_floor(labels: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Same as src.preprocess.apply_rule_floor, vectorized: a flag raises SAFE to SUSPICIOUS."""
    return np.where(flagged & (labels == SAFE), SUSPICIOUS, labels)


def binary_metrics(true_labels, pred_labels: np.ndarray) -> dict:
    y = np.asarray([str(t) != "SAFE" for t in true_labels])
    pred = np.asarray(pred_labels) != SAFE
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sweep(true_labels, classical_probs, guard_probs) -> dict:
    """{rule: {threshold: metrics + escalated share}} plus classical-only and guard-only endpoints."""
    out: dict = {
        "classical_only": binary_metrics(true_labels, np.asarray(classical_probs).argmax(axis=1)),
        "guard_only": binary_metrics(true_labels, np.asarray(guard_probs).argmax(axis=1)),
    }
    for rule in RULES:
        out[rule] = {}
        for t in THRESHOLDS:
            mask = escalate_mask(classical_probs, rule, t)
            out[rule][str(t)] = {**binary_metrics(true_labels, pipeline_labels(classical_probs, guard_probs, mask)),
                                 "escalated_share": float(mask.mean()), "escalated": int(mask.sum())}
    return out


def choose_threshold(validation_sweep: dict, rule: str) -> float:
    """Highest validation binary accuracy; ties broken by fewer escalations, then the lower threshold."""
    rows = [(m["accuracy"], -m["escalated_share"], -float(t), float(t)) for t, m in validation_sweep[rule].items()]
    return max(rows)[3]


# --------------------------------------------------------------------------- latency


def per_text_ms(predict, texts: list[str]) -> np.ndarray:
    for text in texts[:WARMUP]:
        predict([text])
    times = []
    for text in texts:
        start = time.perf_counter()
        predict([text])
        times.append((time.perf_counter() - start) * 1000)
    return np.asarray(times)


def pipeline_latency(classical_ms: np.ndarray, guard_ms: np.ndarray, classical_probs: np.ndarray) -> dict:
    out = {"classical_only_mean_ms": float(classical_ms.mean()), "guard_only_mean_ms": float(guard_ms.mean())}
    for rule in RULES:
        out[rule] = {
            str(t): float((classical_ms + guard_ms * escalate_mask(classical_probs, rule, t)).mean()) for t in THRESHOLDS
        }
    return out


# --------------------------------------------------------------------------- inputs


def _predictors():
    import joblib

    from src.paper_results import ClassicalPredictor, TransformerPredictor

    return ClassicalPredictor(joblib.load(CLASSICAL_PATH)), TransformerPredictor(str(GUARD_DIR))


def _cached_probs(name: str, predict, frame: pd.DataFrame) -> np.ndarray:
    """Probabilities for *frame*, from artifacts/probs/<name>.npz if it matches these texts, else computed and saved."""
    texts = frame["text"].tolist()
    path = probs_io.PROBS_DIR / f"{name}.npz"
    if path.exists():
        try:
            return probs_io.load_probs(path, texts)["probs"]
        except ValueError:
            pass
    probs = np.concatenate([predict(texts[i : i + 64]) for i in range(0, len(texts), 64)])
    probs_io.save_probs(name, probs, frame["label"].tolist(), texts)
    return probs


def _rule_flagged(raw_texts: list[str], cleaned: list[str]) -> np.ndarray:
    from src.preprocess import rule_flags

    return np.asarray([bool(rule_flags(raw, clean)) for raw, clean in zip(raw_texts, cleaned)])


def external_sets() -> list[dict]:
    """External sets whose classical and guard probabilities exist, with raw texts for the rule floor."""
    from src.external_eval import DATASETS, load_external, remove_training_overlap, training_texts
    from src.preprocess import clean_text

    exact, normalized, _ = training_texts()
    out = []
    for ds in DATASETS:
        files = {m: probs_io.PROBS_DIR / f"external_{probs_io.slug(ds.name)}_{m}.npz" for m in ("classical", "guard")}
        if not all(p.exists() for p in files.values()):
            continue
        try:
            frame = load_external(ds)
        except Exception as exc:  # gated or offline: skip, as external_eval does
            print(f"[external] {ds.name} skipped: {type(exc).__name__}: {exc}")
            continue
        kept, _ = remove_training_overlap(frame, exact, normalized)
        cleaned = [clean_text(t) for t in kept["text"]]
        probs = {m: probs_io.load_probs(p, cleaned)["probs"] for m, p in files.items()}
        out.append({"name": ds.name, "labels": kept["label"].tolist(), "probs": probs,
                    "flagged": _rule_flagged(kept["text"].astype(str).tolist(), cleaned)})
    return out


def _guard_hub_matches_seed42() -> dict:
    """Is the Hub guard model (used by external_eval) byte-identical to artifacts/paper_runs/guard/seed_42?"""
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    local = GUARD_DIR / "model.safetensors"
    hub = sorted((Path.home() / ".cache" / "huggingface" / "hub" / "models--Arshu-110405--promptguard-guard").glob(
        "snapshots/*/model.safetensors"))
    if not local.exists() or not hub:
        return {"checked": False, "reason": "local seed_42 weights or Hub cache not found"}
    return {"checked": True, "identical": sha(local) == sha(hub[-1].resolve()), "hub_snapshot": hub[-1].parent.name}


# --------------------------------------------------------------------------- tables


def tables(result: dict) -> dict[str, tuple[list[str], list[list[str]], str]]:
    sel = result["selected_thresholds"]
    rows = []
    for set_name, sw in result["sets"].items():
        lat = result["latency"]
        rows.append([set_name, "classical only", "-", f"{sw['classical_only']['accuracy']:.4f}",
                     f"{sw['classical_only']['false_positive_rate']:.4f}", "0.0", f"{lat['classical_only_mean_ms']:.2f}"])
        for rule in RULES:
            t = str(sel[rule])
            m = sw[rule][t]
            rows.append([set_name, rule.replace("_", " "), t, f"{m['accuracy']:.4f}", f"{m['false_positive_rate']:.4f}",
                         f"{100 * m['escalated_share']:.1f}", f"{lat['classical_only_mean_ms'] + m['escalated_share'] * lat['guard_only_mean_ms']:.2f}"])
        m = sw[RULES[0]][str(DEPLOYED_THRESHOLD)]
        rows.append([set_name, "deployed rule, t = 0.65", str(DEPLOYED_THRESHOLD), f"{m['accuracy']:.4f}",
                     f"{m['false_positive_rate']:.4f}", f"{100 * m['escalated_share']:.1f}",
                     f"{lat['classical_only_mean_ms'] + m['escalated_share'] * lat['guard_only_mean_ms']:.2f}"])
        rows.append([set_name, "guard only", "-", f"{sw['guard_only']['accuracy']:.4f}",
                     f"{sw['guard_only']['false_positive_rate']:.4f}", "100.0", f"{lat['guard_only_mean_ms']:.2f}"])
    escalation = (
        ["evaluation set", "pipeline", "t", "binary accuracy", "FPR", "% escalated", "mean CPU ms"],
        rows,
        "Classical-to-guard escalation. t is chosen on the validation split only (highest binary accuracy). "
        "Mean CPU ms = measured classical time + escalated share x measured guard time (batch size 1, "
        f"{result['config']['latency_samples']} test texts).",
    )
    rf_rows = []
    for set_name, rf in result["rule_floor"].items():
        rf_rows.append([set_name, str(rf["n"]), str(rf["flagged"]), str(rf["labels_changed"]),
                        f"{rf['without']['accuracy']:.4f}", f"{rf['with']['accuracy']:.4f}",
                        f"{rf['without']['false_positive_rate']:.4f}", f"{rf['with']['false_positive_rate']:.4f}"])
    rule_floor = (
        ["evaluation set", "n", "rows flagged", "labels changed", "acc without", "acc with", "FPR without", "FPR with"],
        rf_rows,
        "Effect of the rule floor (chat-template tokens or content hidden in tags raise SAFE to SUSPICIOUS) on the "
        "classical model, binary view.",
    )
    return {"escalation": escalation, "rule_floor": rule_floor}


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latency-samples", type=int, default=LATENCY_SAMPLES)
    args = parser.parse_args(argv)

    from src.paper_results import to_latex, to_markdown
    from src.preprocess import get_splits

    _, val_df, test_df, _ = get_splits()
    classical, guard = _predictors()
    probs = {
        split: {"classical": _cached_probs(f"{split}_classical_deployed", classical.predict, frame),
                "guard": _cached_probs(f"{split}_guard_seed42", guard.predict, frame)}
        for split, frame in (("validation", val_df), ("test", test_df))
    }
    # The deployed classical model should be the paper's classical model (deterministic fit).
    paper_classical = probs_io.PROBS_DIR / "test_classical_seed42.npz"
    deployed_vs_paper = None
    if paper_classical.exists():
        diff = np.abs(probs_io.load_probs(paper_classical, test_df["text"].tolist())["probs"] - probs["test"]["classical"])
        deployed_vs_paper = {"max_abs_prob_difference": float(diff.max())}

    sets, rule_floor = {}, {}
    for split, frame in (("validation", val_df), ("test", test_df)):
        original = (frame["derivation"] == "original").to_numpy()
        name = f"neuralchemy {split} (original rows)"
        labels = frame["label"].to_numpy()[original]
        c, g = probs[split]["classical"][original], probs[split]["guard"][original]
        sets[name] = sweep(labels, c, g)
        flagged = _rule_flagged(frame["raw_text"].tolist(), frame["text"].tolist())[original]
        base = c.argmax(axis=1)
        floored = apply_rule_floor(base, flagged)
        rule_floor[name] = {"n": int(len(labels)), "flagged": int(flagged.sum()), "labels_changed": int((floored != base).sum()),
                            "without": binary_metrics(labels, base), "with": binary_metrics(labels, floored)}
    for ext in external_sets():
        c, g = ext["probs"]["classical"], ext["probs"]["guard"]
        sets[ext["name"]] = sweep(ext["labels"], c, g)
        base = c.argmax(axis=1)
        floored = apply_rule_floor(base, ext["flagged"])
        rule_floor[ext["name"]] = {"n": len(ext["labels"]), "flagged": int(ext["flagged"].sum()),
                                   "labels_changed": int((floored != base).sum()),
                                   "without": binary_metrics(ext["labels"], base), "with": binary_metrics(ext["labels"], floored)}

    val_name = "neuralchemy validation (original rows)"
    selected = {rule: choose_threshold(sets[val_name], rule) for rule in RULES}

    latency_df = test_df.sample(n=min(args.latency_samples, len(test_df)), random_state=0)
    lat_texts = latency_df["text"].tolist()
    classical_ms = per_text_ms(classical.predict, lat_texts)
    guard_ms = per_text_ms(guard.predict, lat_texts)
    latency = pipeline_latency(classical_ms, guard_ms, classical.predict(lat_texts))

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"thresholds": list(THRESHOLDS), "rules": list(RULES), "deployed_rule": "suspicious_only",
                   "deployed_threshold": DEPLOYED_THRESHOLD, "selection": "validation binary accuracy, ties -> fewer escalations",
                   "latency_samples": len(lat_texts), "latency_device": "cpu", "classical": str(CLASSICAL_PATH.relative_to(ROOT)),
                   "guard": GUARD_DIR.relative_to(ROOT).as_posix()},
        "deployed_classical_vs_paper_seed42": deployed_vs_paper,
        "guard_hub_vs_seed42": _guard_hub_matches_seed42(),
        "selected_thresholds": selected,
        "sets": sets,
        "rule_floor": rule_floor,
        "latency": latency,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, (columns, rows, caption) in tables(result).items():
        (TABLES_DIR / f"{name}.md").write_text(f"{caption}\n\n" + to_markdown(columns, rows), encoding="utf-8")
        (TABLES_DIR / f"{name}.tex").write_text(to_latex(columns, rows, caption, f"tab:{name.replace('_', '-')}"),
                                                encoding="utf-8")
    print(to_markdown(*tables(result)["escalation"][:2]))
    print(to_markdown(*tables(result)["rule_floor"][:2]))
    return result


if __name__ == "__main__":
    main()
