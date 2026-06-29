from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    BERT_RUNTIME_AVAILABLE = True
except Exception:
    BERT_RUNTIME_AVAILABLE = False

from src.preprocess import clean_text, highlight_attack_patterns
from src.explain import _merge_subword_scores

import warnings
from src.train_classical import RuleBasedClassifier


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
    def __init__(self, model_type: str = "classical", escalate_on_uncertain: bool = False) -> None:
        self.model_type = model_type.lower()
        root = Path(__file__).resolve().parents[1]
        models_dir = root / "models"
        if self.model_type not in {"classical", "bert", "guard"}:
            raise ValueError("model_type must be 'classical', 'bert', or 'guard'.")

        self.classical_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.guard_tokenizer = None
        self.guard_model = None
        self.escalate_on_uncertain = escalate_on_uncertain
        self.is_rule_based_fallback = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if BERT_RUNTIME_AVAILABLE else None

        if self.model_type == "classical":
            model_path = models_dir / "classical_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Classical model missing at {model_path}.")
            self.classical_model = joblib.load(model_path)
            self.is_rule_based_fallback = isinstance(self.classical_model, RuleBasedClassifier)
            if self.is_rule_based_fallback:
                warnings.warn(
                    "Loaded model is RuleBasedClassifier (keyword-only fallback), not a "
                    "trained ML pipeline. Re-run src/train_classical.py with scikit-learn "
                    "installed to fix this. Predictions will be inaccurate.",
                    stacklevel=2,
                )
            if self.escalate_on_uncertain and BERT_RUNTIME_AVAILABLE:
                guard_dir = models_dir / "guard_model"
                if guard_dir.exists():
                    self.guard_tokenizer = AutoTokenizer.from_pretrained(str(guard_dir))
                    self.guard_model = AutoModelForSequenceClassification.from_pretrained(str(guard_dir))
                    self.guard_model.to(self.device)
                    self.guard_model.eval()
        elif self.model_type == "bert":
            if not BERT_RUNTIME_AVAILABLE:
                raise RuntimeError("BERT runtime unavailable due to blocked dependencies on this environment.")
            bert_dir = models_dir / "bert_model"
            if not bert_dir.exists():
                raise FileNotFoundError(f"BERT model missing at {bert_dir}.")
            self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained(str(bert_dir))
            self.bert_model = DistilBertForSequenceClassification.from_pretrained(
                str(bert_dir), output_attentions=True
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
        else:
            if not BERT_RUNTIME_AVAILABLE:
                raise RuntimeError("Guard model runtime unavailable due to blocked dependencies on this environment.")
            guard_dir = models_dir / "guard_model"
            if not guard_dir.exists():
                raise FileNotFoundError(f"Guard model missing at {guard_dir}.")
            self.guard_tokenizer = AutoTokenizer.from_pretrained(str(guard_dir))
            self.guard_model = AutoModelForSequenceClassification.from_pretrained(str(guard_dir))
            self.guard_model.to(self.device)
            self.guard_model.eval()

    def _predict_classical(self, text: str) -> tuple[str, float]:
        if self.classical_model is None:
            raise ValueError("Classical model is not loaded.")
        probs = self.classical_model.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return str(self.classical_model.classes_[idx]), float(probs[idx])

    def _predict_bert(self, text: str) -> tuple[str, float]:
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model is not loaded.")
        if not BERT_RUNTIME_AVAILABLE or self.device is None:
            raise RuntimeError("BERT runtime unavailable.")
        encoded = self.bert_tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.bert_model(**encoded).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    def _predict_guard(self, text: str) -> tuple[str, float]:
        if self.guard_model is None or self.guard_tokenizer is None:
            raise ValueError("Guard model is not loaded.")
        if not BERT_RUNTIME_AVAILABLE or self.device is None:
            raise RuntimeError("Guard model runtime unavailable.")
        encoded = self.guard_tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.guard_model(**encoded).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    def _get_token_importance(self, text: str) -> dict[str, float]:
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model is not loaded.")
        if not BERT_RUNTIME_AVAILABLE or self.device is None:
            return {}
        encoded = self.bert_tokenizer(
            text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.bert_model(**encoded, output_attentions=True)
        last_attention = outputs.attentions[-1].mean(dim=1).squeeze(0)
        token_scores = last_attention.mean(dim=0).detach().cpu().numpy()
        tokens = self.bert_tokenizer.convert_ids_to_tokens(
            encoded["input_ids"].squeeze(0).detach().cpu().numpy()
        )

        raw_scores = _merge_subword_scores(list(tokens), [float(s) for s in token_scores])
        if not raw_scores:
            return {}
        values = np.array(list(raw_scores.values()))
        min_v = float(values.min())
        max_v = float(values.max())
        denom = max(max_v - min_v, 1e-8)
        return {k: (v - min_v) / denom for k, v in raw_scores.items()}

    def predict(self, text: str) -> dict:
        normalized = clean_text(text)
        patterns = highlight_attack_patterns(normalized, already_clean=True)
        if self.model_type == "classical":
            label, confidence = self._predict_classical(normalized)
        elif self.model_type == "bert":
            label, confidence = self._predict_bert(normalized)
        else:
            label, confidence = self._predict_guard(normalized)

        escalated = False
        if (
            self.escalate_on_uncertain
            and self.model_type == "classical"
            and label == "SUSPICIOUS"
            and confidence < 0.65
            and self.guard_model is not None
        ):
            label, confidence = self._predict_guard(normalized)
            escalated = True

        token_highlights = self._get_token_importance(normalized) if self.model_type == "bert" else {}
        base_risk = {"SAFE": 15.0, "SUSPICIOUS": 55.0, "MALICIOUS": 85.0}[label]
        pattern_boost = min(len(patterns) * 4.0, 15.0)
        risk_score = max(0.0, min(100.0, base_risk + pattern_boost + (confidence - 0.5) * 20))
        explanation = _explain_label(label, patterns, confidence)
        if self.is_rule_based_fallback:
            explanation += (
                " WARNING: keyword-only fallback model active — "
                "confidence values are not meaningful."
            )

        return {
            "label": label,
            "confidence": confidence,
            "risk_score": risk_score,
            "attack_patterns": patterns,
            "token_highlights": token_highlights,
            "explanation": explanation,
            "is_rule_based_fallback": self.is_rule_based_fallback,
            "escalated": escalated,
        }

    def batch_predict(self, texts: list[str]) -> list[dict]:
        return [self.predict(text) for text in texts]
