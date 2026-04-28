from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

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


def get_token_importance(text: str) -> dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    bert_dir = root / "models" / "bert_model"
    if not bert_dir.exists():
        raise FileNotFoundError(f"BERT model directory not found at {bert_dir}.")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(bert_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(bert_dir), output_attentions=True)
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

    raw_scores = {tok: float(score) for tok, score in zip(tokens, token_scores) if tok not in {"[PAD]"}}
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
        key = word.lower()
        score = scores.get(key, scores.get(f"##{key}", 0.0))
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
