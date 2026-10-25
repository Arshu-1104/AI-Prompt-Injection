"""Untrained baselines for the paper, evaluated exactly like the trained models.

Each factory returns a predictor whose .predict(texts) gives (n, 3) probabilities
in LABELS order (texts are already clean_text-normalized, as the API sees them):

* regex     - flagged MALICIOUS if any pattern of the explanation layer
              (src.preprocess.ATTACK_PATTERNS / CHAT_TEMPLATE_PATTERNS) matches.
              These patterns were written by the authors, so in-distribution this
              baseline may be optimistic.
* keyword   - the fixed keyword lists of src.train_classical.RuleBasedClassifier
              (suspicious terms -> SUSPICIOUS, malicious terms -> MALICIOUS).
* llama_pg2 - meta-llama/Llama-Prompt-Guard-2-86M as published (gated: accept the
              licence on the Hub and set HF_TOKEN). Its two labels map to SAFE and
              MALICIOUS like the ProtectAI baseline, so it never predicts SUSPICIOUS.
"""
from __future__ import annotations

import numpy as np

from src.metrics import LABELS

LLAMA_PG2_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
# Label names Llama Prompt Guard 2 may use; the loaded config's id2label is recorded in the results.
LLAMA_PG2_LABELS = {"LABEL_0": "SAFE", "LABEL_1": "MALICIOUS", "BENIGN": "SAFE", "MALICIOUS": "MALICIOUS"}
BASELINES = ("regex", "keyword", "llama_pg2")


def _one_hot(labels: list[str]) -> np.ndarray:
    out = np.zeros((len(labels), len(LABELS)))
    out[np.arange(len(labels)), [LABELS.index(label) for label in labels]] = 1.0
    return out


class RegexBaseline:
    source = "src.preprocess.ATTACK_PATTERNS + CHAT_TEMPLATE_PATTERNS (any match = MALICIOUS)"

    def predict(self, texts: list[str]) -> np.ndarray:
        from src.preprocess import highlight_attack_patterns

        return _one_hot(["MALICIOUS" if highlight_attack_patterns(t, already_clean=True) else "SAFE" for t in texts])


class KeywordBaseline:
    source = "src.train_classical.RuleBasedClassifier keyword lists"

    def __init__(self) -> None:
        from src.train_classical import RuleBasedClassifier

        self.classifier = RuleBasedClassifier()

    def predict(self, texts: list[str]) -> np.ndarray:
        probs = np.asarray(self.classifier.predict_proba(list(texts)), dtype=float)
        classes = [str(c) for c in self.classifier.classes_]
        return probs[:, [classes.index(label) for label in LABELS]]


def llama_pg2(device: str = "cpu"):
    from src.paper_results import TransformerPredictor

    return TransformerPredictor(LLAMA_PG2_ID, max_length=512, label_map=LLAMA_PG2_LABELS, device=device)


FACTORIES = {"regex": RegexBaseline, "keyword": KeywordBaseline, "llama_pg2": llama_pg2}
