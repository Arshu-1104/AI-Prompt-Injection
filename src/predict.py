from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    BERT_RUNTIME_AVAILABLE = True
    BERT_RUNTIME_ERROR = ""
except Exception as _exc:  # ImportError, or OSError when a native DLL is blocked
    BERT_RUNTIME_AVAILABLE = False
    BERT_RUNTIME_ERROR = f"torch/transformers not importable ({type(_exc).__name__}: {_exc})"

from src.model_sources import DownloadFailedError, RuntimeUnavailableError, resolve_model_source
from src.preprocess import (
    apply_rule_floor,
    clean_text,
    empty_after_cleaning_label,
    highlight_attack_patterns,
    rule_flags,
)
from src.explain import _merge_subword_scores
from src.train_classical import RuleBasedClassifier


LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]
MAX_LENGTH = 256  # fallback for weights saved before promptguard_max_length existed


def model_max_length(model) -> int:
    """The --max-length the model was trained with (stored in its config)."""
    return int(getattr(model.config, "promptguard_max_length", MAX_LENGTH))


def _load_transformer(model_type: str, device, **model_kwargs):
    """Load (tokenizer, model) for 'bert' or 'guard' from the Hub or models/."""
    if not BERT_RUNTIME_AVAILABLE:
        raise RuntimeUnavailableError(BERT_RUNTIME_ERROR)
    source = resolve_model_source(model_type)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source, **model_kwargs)
    except OSError as exc:
        if Path(source).is_dir():
            raise
        # Hub repo id: unreachable Hub, missing/private repo, or bad token.
        raise DownloadFailedError(f"Could not fetch {source} from the Hugging Face Hub: {exc}") from exc
    model.to(device)
    model.eval()
    return tokenizer, model, source


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
    def __init__(
        self,
        model_type: str = "classical",
        escalate_on_uncertain: bool = False,
        escalation_analyzer: "PromptAnalyzer | None" = None,
        device: str | None = None,
    ) -> None:
        """*escalation_analyzer*: an already-loaded guard analyzer whose model the
        classical analyzer reuses for escalation instead of loading a second copy.
        *device*: torch device for transformer models (default: cuda if available)."""
        self.model_type = model_type.lower()
        models_dir = Path(__file__).resolve().parents[1] / "models"
        if self.model_type not in {"classical", "bert", "guard"}:
            raise ValueError("model_type must be 'classical', 'bert', or 'guard'.")

        self.classical_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.guard_tokenizer = None
        self.guard_model = None
        self.escalate_on_uncertain = escalate_on_uncertain
        self.is_rule_based_fallback = False
        self.source = None
        self.device = None
        if BERT_RUNTIME_AVAILABLE:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

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
            self.source = str(model_path)
            if escalation_analyzer is not None:
                self.guard_tokenizer = escalation_analyzer.guard_tokenizer
                self.guard_model = escalation_analyzer.guard_model
            elif self.escalate_on_uncertain and BERT_RUNTIME_AVAILABLE:
                # Escalation is optional: classical must still load without guard weights.
                try:
                    self.guard_tokenizer, self.guard_model, _ = _load_transformer("guard", self.device)
                except Exception as exc:
                    warnings.warn(f"Guard escalation disabled: {exc}", stacklevel=2)
        elif self.model_type == "bert":
            self.bert_tokenizer, self.bert_model, self.source = _load_transformer(
                "bert", self.device, output_attentions=True
            )
        else:
            self.guard_tokenizer, self.guard_model, self.source = _load_transformer("guard", self.device)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """(n, 3) class probabilities in LABELS order for already-normalized texts."""
        texts = list(texts)
        if self.model_type == "classical":
            probs = np.asarray(self.classical_model.predict_proba(texts), dtype=float)
            classes = [str(c) for c in self.classical_model.classes_]
            return probs[:, [classes.index(label) for label in LABELS]]
        model, tokenizer = (
            (self.bert_model, self.bert_tokenizer) if self.model_type == "bert" else (self.guard_model, self.guard_tokenizer)
        )
        encoded = tokenizer(texts, truncation=True, max_length=model_max_length(model), padding=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().numpy()
        names = [str(model.config.id2label[i]) for i in range(probs.shape[1])]
        return probs[:, [names.index(label) for label in LABELS]]

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
            text, truncation=True, max_length=model_max_length(self.bert_model), return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.bert_model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return str(self.bert_model.config.id2label[idx]), float(probs[idx])

    def _predict_guard(self, text: str) -> tuple[str, float]:
        if self.guard_model is None or self.guard_tokenizer is None:
            raise ValueError("Guard model is not loaded.")
        if not BERT_RUNTIME_AVAILABLE or self.device is None:
            raise RuntimeError("Guard model runtime unavailable.")
        encoded = self.guard_tokenizer(
            text, truncation=True, max_length=model_max_length(self.guard_model), return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.guard_model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        return str(self.guard_model.config.id2label[idx]), float(probs[idx])

    def _get_token_importance(self, text: str) -> dict[str, float]:
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model is not loaded.")
        if not BERT_RUNTIME_AVAILABLE or self.device is None:
            return {}
        encoded = self.bert_tokenizer(
            text, truncation=True, max_length=model_max_length(self.bert_model), return_tensors="pt"
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
        # Patterns and rules see the raw text: cleaning strips tag-shaped payloads.
        patterns = highlight_attack_patterns(normalized, already_clean=True, raw_text=text)
        flags = rule_flags(text, normalized)
        empty_input = not normalized.strip()
        if empty_input:
            # Nothing left for a model to judge; the rule layer decides.
            label, confidence = empty_after_cleaning_label(flags), 1.0
        elif self.model_type == "classical":
            label, confidence = self._predict_classical(normalized)
        elif self.model_type == "bert":
            label, confidence = self._predict_bert(normalized)
        else:
            label, confidence = self._predict_guard(normalized)

        escalated = False
        if (
            not empty_input
            and self.escalate_on_uncertain
            and self.model_type == "classical"
            and label == "SUSPICIOUS"
            and confidence < 0.65
            and self.guard_model is not None
        ):
            label, confidence = self._predict_guard(normalized)
            escalated = True

        label, raised_by_rule = apply_rule_floor(label, flags)

        token_highlights = (
            self._get_token_importance(normalized) if self.model_type == "bert" and not empty_input else {}
        )
        base_risk = {"SAFE": 15.0, "SUSPICIOUS": 55.0, "MALICIOUS": 85.0}[label]
        pattern_boost = min(len(patterns) * 4.0, 15.0)
        risk_score = max(0.0, min(100.0, base_risk + pattern_boost + (confidence - 0.5) * 20))
        explanation = _explain_label(label, patterns, confidence)
        if empty_input:
            explanation += (
                " No text is left after normalization, so no model was run; "
                f"the label comes from the rule layer ({', '.join(flags) or 'no rule matched'})."
            )
        if raised_by_rule:
            explanation += (
                f" Raised from SAFE to SUSPICIOUS by the rule layer ({', '.join(flags)}); "
                "confidence is the model's confidence in its own SAFE prediction."
            )
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
