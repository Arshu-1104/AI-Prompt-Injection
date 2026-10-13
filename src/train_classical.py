from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when sklearn is broken/missing
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc

from src.metrics import evaluate_split
from src.preprocess import get_splits

logger = logging.getLogger(__name__)


_FALLBACK_BANNER = (
    "\n"
    "############################################################\n"
    "#  scikit-learn is NOT available in this environment.     #\n"
    "#  Training will fall back to RuleBasedClassifier, a      #\n"
    "#  keyword-only stub. Its predictions are NOT a real ML   #\n"
    "#  model and should never be used in production.          #\n"
    "############################################################\n"
)

# Augment training data with realistic short/everyday benign inputs.
# The source dataset's SAFE class skews long and templated ("Generate a X
# for Y"), with no true short-form conversational text — so the model never
# learned what an ordinary greeting or simple question looks like, and
# short inputs like "Hello" default toward the MALICIOUS majority signal.
EVERYDAY_SAFE_EXAMPLES = [
    "Hello", "Hi there", "Hey, how are you?", "Good morning", "Thanks so much",
    "Thank you for your help", "What's the weather like today?", "How do I reset my password?",
    "Can you help me write a Python function?", "What's the capital of France?",
    "Can you summarize this article for me?", "What time is it in Tokyo?",
    "Give me a recipe for pasta", "How do I improve my resume?",
    "Explain quantum computing simply", "What's a good book to read?",
    "Can you help me debug this code?", "How do plants photosynthesize?",
    "What's the square root of 144?", "Tell me a fun fact about space",
    "How do I say hello in French?", "What's the best way to learn guitar?",
    "Can you proofread this email?", "What's 15% of 200?",
    "How's your day going?", "See you later", "Sounds good, thanks",
    "Can you explain this concept to me?", "I need help with my homework",
    "What's a healthy breakfast option?",
]


class RuleBasedClassifier:
    """Keyword-matching stub. Only ever used when sklearn is unavailable AND the
    caller has explicitly opted in via allow_rule_based_fallback=True. Not a
    substitute for a trained model — coverage is limited to the phrases below.
    """

    classes_ = np.array(["SAFE", "SUSPICIOUS", "MALICIOUS"])

    suspicious_terms = [
        "ignore previous",
        "ignore all previous",
        "ignore prior",
        "disregard",
        "pretend",
        "act as if",
        "new task",
        "forget your rules",
        "forget the rules",
        "new instructions",
        "role play",
        "roleplay",
        "hypothetically",
        "for research purposes",
        "as an ai without",
    ]
    malicious_terms = [
        "jailbreak",
        "bypass",
        "system prompt",
        "leak",
        "override policy",
        "override your",
        "dan mode",
        "developer mode",
        "no restrictions",
        "without restrictions",
        "reveal your instructions",
        "reveal the system prompt",
        "exfiltrate",
        "extract the api key",
        "print your instructions",
    ]

    def fit(self, x_train: Any, y_train: Any) -> "RuleBasedClassifier":
        _ = (x_train, y_train)
        return self

    def predict_proba(self, texts: list[str] | Any) -> np.ndarray:
        probs: list[list[float]] = []
        for text in list(texts):
            t = str(text).lower()
            s_hits = sum(term in t for term in self.suspicious_terms)
            m_hits = sum(term in t for term in self.malicious_terms)
            if m_hits > 0:
                probs.append([0.05, 0.15, 0.80])
            elif s_hits > 0:
                probs.append([0.20, 0.70, 0.10])
            else:
                probs.append([0.90, 0.08, 0.02])
        return np.array(probs, dtype=float)

    def predict(self, texts: list[str] | Any) -> np.ndarray:
        probs = self.predict_proba(texts)
        return self.classes_[np.argmax(probs, axis=1)]


def _word_tfidf() -> "TfidfVectorizer":
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True)


def _word_char_tfidf() -> "FeatureUnion":
    # Character n-grams within word boundaries add robustness to spelling,
    # spacing and leetspeak changes that word n-grams miss.
    return FeatureUnion(
        [
            ("word", _word_tfidf()),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)),
        ]
    )


# Inverse regularization strengths tried for logistic regression.
LOGREG_C_GRID = (0.3, 1.0, 3.0, 10.0)


def candidate_pipelines(random_state: int) -> dict[str, "Pipeline"]:
    """The fixed candidate set; the winner is chosen on the validation split only.

    class_weight="balanced": the training data is skewed toward MALICIOUS
    (~60/40 in the source plus leetspeak variants) with the synthetic SUSPICIOUS
    class as a thin minority. MultinomialNB has no class_weight, so it gets an
    explicit uniform class_prior for the same reason.
    """
    candidates: dict[str, Pipeline] = {}
    for features_name, features in (("word", _word_tfidf), ("word+char", _word_char_tfidf)):
        for c in LOGREG_C_GRID:
            candidates[f"logreg[{features_name},C={c:g}]"] = Pipeline(
                steps=[
                    ("tfidf", features()),
                    ("classifier", LogisticRegression(max_iter=2000, C=c, random_state=random_state, class_weight="balanced")),
                ]
            )
    candidates["random_forest"] = Pipeline(
        steps=[
            ("tfidf", _word_tfidf()),
            ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, class_weight="balanced")),
        ]
    )
    candidates["multinomial_nb"] = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=False)),
            ("classifier", MultinomialNB(class_prior=[1 / 3, 1 / 3, 1 / 3])),
        ]
    )
    return candidates


def fit_classical(random_state: int = 42, allow_rule_based_fallback: bool = False) -> dict:
    """Fit candidates on the official train split, pick the best by weighted F1 on
    the official validation split, and report it on the test split. Saves nothing.

    *random_state* seeds the classifiers only; the splits are the Hub's own.
    Returns pipeline, best_name, validation (per candidate), test (3-class and
    binary, see src.metrics.evaluate_split) and dataset (provenance).

    Raises RuntimeError if scikit-learn is unavailable, unless the caller
    explicitly passes allow_rule_based_fallback=True (e.g. for a quick local
    smoke test). This prevents silently shipping the keyword-only stub as if
    it were a trained model.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning(_FALLBACK_BANNER)
        print(_FALLBACK_BANNER, file=sys.stderr)
        print(f"Import error was: {SKLEARN_IMPORT_ERROR!r}", file=sys.stderr)
        if not allow_rule_based_fallback:
            raise RuntimeError(
                "scikit-learn is not importable in this environment, so "
                "train_classical() refused to silently fall back to the "
                "keyword-only RuleBasedClassifier. Fix your environment "
                "(e.g. `pip install -r requirements.txt` inside the active "
                "venv/container) and retrain. If you really want the "
                "keyword-only stub for a quick offline test, call "
                "train_classical(allow_rule_based_fallback=True) explicitly."
            ) from SKLEARN_IMPORT_ERROR

    train_df, val_df, test_df, provenance = get_splits()

    # Everyday SAFE examples go into the training set only.
    everyday = EVERYDAY_SAFE_EXAMPLES * 8
    x_train = pd.concat([train_df["text"], pd.Series(everyday)], ignore_index=True)
    y_train = pd.concat([train_df["label"], pd.Series(["SAFE"] * len(everyday))], ignore_index=True)

    if SKLEARN_AVAILABLE:
        validation: dict[str, dict] = {}
        pipelines: dict[str, Pipeline] = {}
        for name, pipeline in candidate_pipelines(random_state).items():
            pipeline.fit(x_train, y_train)
            validation[name] = evaluate_split(val_df, _ordered_proba(pipeline, val_df["text"]))
            pipelines[name] = pipeline
            print(f"  candidate {name:<28} validation 3-class F1_w={validation[name]['three_class']['f1_weighted']:.4f}")
        best_name = max(validation, key=lambda m: validation[m]["three_class"]["f1_weighted"])
        best_pipeline = pipelines[best_name]
    else:
        best_name = "rule_based_fallback"
        best_pipeline = RuleBasedClassifier().fit(x_train.tolist(), y_train.tolist())
        validation = {best_name: evaluate_split(val_df, _ordered_proba(best_pipeline, val_df["text"]))}

    return {
        "pipeline": best_pipeline,
        "best_name": best_name,
        "selection": "best weighted F1 on the official validation split",
        "validation": validation,
        "test": evaluate_split(test_df, _ordered_proba(best_pipeline, test_df["text"])),
        "dataset": provenance,
    }


def _ordered_proba(model, texts) -> np.ndarray:
    """predict_proba with columns in src.metrics.LABELS order."""
    from src.metrics import LABELS

    probs = np.asarray(model.predict_proba(list(texts)), dtype=float)
    classes = [str(c) for c in model.classes_]
    return probs[:, [classes.index(label) for label in LABELS]]


def train_classical(random_state: int = 42, allow_rule_based_fallback: bool = False) -> dict:
    """Train the classical model and save it to models/classical_model.pkl."""
    fitted = fit_classical(random_state, allow_rule_based_fallback)
    best_pipeline, best_name = fitted["pipeline"], fitted["best_name"]

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    artifacts_dir = root / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, models_dir / "classical_model.pkl")
    if not isinstance(best_pipeline, RuleBasedClassifier):
        fitted_vectorizer = best_pipeline.named_steps["tfidf"]
        joblib.dump(fitted_vectorizer, models_dir / "tfidf_vectorizer.pkl")

    metrics_path = artifacts_dir / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    metrics_data["classical_model"] = {
        "best_model": best_name,
        "random_state": random_state,
        "selection": fitted["selection"],
        "validation": fitted["validation"],
        "test": fitted["test"],
        "dataset": fitted["dataset"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    test = fitted["test"]
    print(f"Best classical model (by validation weighted F1): {best_name}")
    print(
        f"Test 3-class: accuracy={test['three_class']['accuracy']:.4f} f1_weighted={test['three_class']['f1_weighted']:.4f} "
        f"ece={test['three_class']['ece']:.4f}"
    )
    print(
        f"Test binary (original Hub rows): accuracy={test['binary']['accuracy']:.4f} f1={test['binary']['f1']:.4f} "
        f"fpr={test['binary']['false_positive_rate']:.4f} ece={test['binary']['ece']:.4f}"
    )
    print(f"Saved classical model to {models_dir / 'classical_model.pkl'}")
    return metrics_data["classical_model"]


def main() -> None:
    train_classical()


if __name__ == "__main__":
    main()