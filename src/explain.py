from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import numpy as np
try:
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    BERT_EXPLAIN_AVAILABLE = True
except Exception:
    BERT_EXPLAIN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except Exception as exc:
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)


LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]


def _extract_classical_top_features() -> None:
    if not PLOTTING_AVAILABLE:
        print(f"Plotting unavailable due to environment policy: {PLOTTING_IMPORT_ERROR}")
        return

    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "classical_model.pkl"
    figures_dir = root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Classical model not found at {model_path}.")

    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.array(tfidf.get_feature_names_out())

    if hasattr(classifier, "coef_"):
        weights = classifier.coef_
        for idx, label in enumerate(LABELS):
            class_weights = weights[idx]
            top_idx = np.argsort(class_weights)[-20:]
            top_features = feature_names[top_idx]
            top_values = class_weights[top_idx]
            order = np.argsort(top_values)

            plt.figure(figsize=(8, 6))
            plt.barh(top_features[order], top_values[order], color="#4C72B0")
            plt.title(f"Top TF-IDF Features - {label}")
            plt.xlabel("Weight")
            plt.tight_layout()
            plt.savefig(figures_dir / f"tfidf_features_{label}.png", dpi=140)
            plt.close()
    elif hasattr(classifier, "feature_importances_"):
        global_importance = classifier.feature_importances_
        top_idx = np.argsort(global_importance)[-20:]
        top_features = feature_names[top_idx]
        top_values = global_importance[top_idx]
        order = np.argsort(top_values)

        plt.figure(figsize=(8, 6))
        plt.barh(top_features[order], top_values[order], color="#4C72B0")
        plt.title("Top Feature Importances (Random Forest - Global)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(figures_dir / "tfidf_features_RF_global.png", dpi=140)
        plt.close()
    else:
        raise ValueError("Unsupported classical classifier for feature explanation.")


def _merge_subword_scores(tokens: list[str], scores: list[float]) -> dict[str, float]:
    """Merge DistilBERT subword tokens (## prefixed) back into full words.

    Consecutive subword pieces are joined and their attention scores are averaged,
    producing a dict keyed by full lowercase words rather than subword tokens.
    Special tokens ([CLS], [SEP], [PAD]) are discarded.
    """
    word_scores: dict[str, float] = {}
    current_word = ""
    current_scores: list[float] = []

    def _flush() -> None:
        if current_word:
            merged = current_word.lower()
            avg = float(np.mean(current_scores)) if current_scores else 0.0
            # Accumulate — if word appears multiple times take the max score
            if merged in word_scores:
                word_scores[merged] = max(word_scores[merged], avg)
            else:
                word_scores[merged] = avg

    for tok, score in zip(tokens, scores):
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            _flush()
            current_word = ""
            current_scores = []
            continue
        if tok.startswith("##"):
            # Continuation of the previous word
            current_word += tok[2:]
            current_scores.append(score)
        else:
            _flush()
            current_word = tok
            current_scores = [score]

    _flush()
    return word_scores


WORD_TOKEN = r"(?u)\b\w\w+\b"  # TfidfVectorizer's default token_pattern


def classical_word_scores(pipeline, text: str, label: str, top_k: int = 10) -> dict[str, float]:
    """Words that push the TF-IDF + logistic-regression pipeline towards *label*.

    Exact per-feature contribution to the label's logit, centred over classes (which
    leaves the softmax unchanged): x_f * (w[label, f] - mean_c w[c, f]). Each word
    n-gram's contribution is split evenly over its words, and each char_wb n-gram's
    over the words of *text* it occurs in; that split to words is an approximation.
    Returns the top_k words with a positive score, scaled to (0, 1] by the largest.
    Works for the deployed pipeline (FeatureUnion of 'word' and 'char' vectorizers);
    returns {} for any other model.
    """
    import re

    try:
        union = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]
        names = union.get_feature_names_out()
        coef = np.asarray(classifier.coef_)
    except (AttributeError, KeyError):
        return {}
    classes = [str(c) for c in classifier.classes_]
    if label not in classes or coef.shape[0] != len(classes):
        return {}
    weights = coef[classes.index(label)] - coef.mean(axis=0)
    x = union.transform([text]).tocoo()
    words = re.findall(WORD_TOKEN, text.lower())
    if not words:
        return {}
    scores: dict[str, float] = {}
    for col, value in zip(x.col, x.data):
        contribution = float(value * weights[col])
        kind, _, feature = names[col].partition("__")
        if kind == "word":
            owners = feature.split()
        else:  # char_wb n-gram: the words of this text that contain it (padded with spaces, as char_wb does)
            owners = [w for w in set(words) if feature in f" {w} "]
        for owner in owners:
            scores[owner] = scores.get(owner, 0.0) + contribution / len(owners)
    positive = sorted(((w, s) for w, s in scores.items() if s > 0), key=lambda item: item[1], reverse=True)[:top_k]
    if not positive:
        return {}
    top = positive[0][1]
    return {w: s / top for w, s in positive}


def get_token_importance(text: str) -> dict[str, float]:
    if not BERT_EXPLAIN_AVAILABLE:
        return {}
    from src.model_sources import resolve_model_source

    source = resolve_model_source("bert")  # BERT_MODEL_ID, else models/bert_model
    tokenizer = DistilBertTokenizerFast.from_pretrained(source)
    model = DistilBertForSequenceClassification.from_pretrained(source, output_attentions=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
    last_attention = outputs.attentions[-1].mean(dim=1).squeeze(0)  # [seq, seq]
    token_scores = last_attention.mean(dim=0).detach().cpu().numpy()
    token_ids = encoded["input_ids"].squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Merge subword pieces → full-word scores
    raw_scores = _merge_subword_scores(list(tokens), [float(s) for s in token_scores])
    if not raw_scores:
        return {}
    values = np.array(list(raw_scores.values()))
    min_v, max_v = float(values.min()), float(values.max())
    denom = max(max_v - min_v, 1e-8)
    normalized = {k: (v - min_v) / denom for k, v in raw_scores.items()}
    return normalized


def highlight_risky_tokens(text: str, scores: dict[str, float]) -> str:
    words = text.split()
    html_parts: list[str] = []
    for word in words:
        # Strip common punctuation before lookup so "instructions," hits "instructions"
        key = word.lower().strip(".,!?;:\"'()")
        score = scores.get(key, 0.0)
        if score >= 0.66:
            color = "#ff6b6b"
        elif score >= 0.33:
            color = "#ffd166"
        else:
            color = "#7bd389"
        html_parts.append(
            f"<span style='background-color:{color};padding:2px 4px;border-radius:4px;margin:1px;'>"
            f"{escape(word)}</span>"
        )
    return " ".join(html_parts)


def main() -> None:
    _extract_classical_top_features()
    print("Saved classical TF-IDF feature charts.")


if __name__ == "__main__":
    main()
