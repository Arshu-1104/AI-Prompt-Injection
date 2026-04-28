from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.explain import get_token_importance
from src.preprocess import clean_text, highlight_attack_patterns


LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]


def _explain_label(label: str, patterns: list[str], confidence: float) -> str:
    if label == "MALICIOUS":
        return (
            f"High-risk prompt detected with confidence {confidence:.2f}. "
            f"Matched patterns: {', '.join(patterns) if patterns else 'none'}."
        )
    if label == "SUSPICIOUS":
        return (
            f"Potential manipulation behavior detected with confidence {confidence:.2f}. "
            "Review and sanitize before sending to an LLM."
        )
    return (
        f"Prompt appears safe with confidence {confidence:.2f}. "
        "No strong injection indicators were detected."
    )


class PromptAnalyzer:
    def __init__(self, model_type: str = "classical") -> None:
        self.model_type = model_type.lower()
        root = Path(__file__).resolve().parents[1]
        models_dir = root / "models"
        if self.model_type not in {"classical", "bert"}:
            raise ValueError("model_type must be 'classical' or 'bert'.")

        self.classical_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "classical":
            model_path = models_dir / "classical_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Classical model missing at {model_path}.")
            self.classical_model = joblib.load(model_path)
        else:
            bert_dir = models_dir / "bert_model"
            if not bert_dir.exists():
                raise FileNotFoundError(f"BERT model missing at {bert_dir}.")
            self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained(str(bert_dir))
            self.bert_model = DistilBertForSequenceClassification.from_pretrained(str(bert_dir))
            self.bert_model.to(self.device)
            self.bert_model.eval()

    def _predict_classical(self, text: str) -> tuple[str, float]:
        if self.classical_model is None:
            raise ValueError("Classical model is not loaded.")
        probs = self.classical_model.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return str(self.classical_model.classes_[idx]), float(probs[idx])

    def _predict_bert(self, text: str) -> tuple[str, float]:
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model is not loaded.")
        encoded = self.bert_tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.bert_model(**encoded).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    def predict(self, text: str) -> dict:
        normalized = clean_text(text)
        patterns = highlight_attack_patterns(normalized)
        if self.model_type == "classical":
            label, confidence = self._predict_classical(normalized)
        else:
            label, confidence = self._predict_bert(normalized)

        token_highlights = get_token_importance(normalized) if self.model_type == "bert" else {}
        base_risk = {"SAFE": 15.0, "SUSPICIOUS": 55.0, "MALICIOUS": 85.0}[label]
        pattern_boost = min(len(patterns) * 4.0, 15.0)
        risk_score = max(0.0, min(100.0, base_risk + pattern_boost + (confidence - 0.5) * 20))
        explanation = _explain_label(label, patterns, confidence)

        return {
            "label": label,
            "confidence": confidence,
            "risk_score": risk_score,
            "attack_patterns": patterns,
            "token_highlights": token_highlights,
            "explanation": explanation,
        }

    def batch_predict(self, texts: list[str]) -> list[dict]:
        return [self.predict(text) for text in texts]
