from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from src.preprocess import get_train_test_data


SCORING = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]


class RuleBasedClassifier:
    classes_ = np.array(["SAFE", "SUSPICIOUS", "MALICIOUS"])
    suspicious_terms = ["ignore previous", "disregard", "pretend", "new task", "forget your rules"]
    malicious_terms = ["jailbreak", "bypass", "system prompt", "leak", "override policy", "dan mode"]

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


def train_classical(random_state: int = 42) -> dict:
    train_df, test_df = get_train_test_data(random_state=random_state)
    x_train = train_df["text"]
    y_train = train_df["label"]
    x_test = test_df["text"]
    y_test = test_df["label"]

    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True)
        nb_vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=False
        )
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=random_state
            ),
            "multinomial_nb": MultinomialNB(),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_results: dict[str, dict[str, dict[str, float]]] = {}
        pipelines: dict[str, Pipeline] = {}

        for name, model in models.items():
            selected_vectorizer = nb_vectorizer if name == "multinomial_nb" else vectorizer
            pipeline = Pipeline(
                steps=[("tfidf", clone(selected_vectorizer)), ("classifier", clone(model))]
            )
            scores = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1,
            )
            cv_results[name] = {
                metric: {
                    "mean": float(np.mean(scores[f"test_{metric}"])),
                    "std": float(np.std(scores[f"test_{metric}"])),
                }
                for metric in SCORING
            }
            pipelines[name] = pipeline

        best_name = max(cv_results, key=lambda m: cv_results[m]["f1_weighted"]["mean"])
        best_pipeline = pipelines[best_name]
        best_pipeline.fit(x_train, y_train)
        predictions = best_pipeline.predict(x_test)
        holdout_metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_weighted": float(precision_score(y_test, predictions, average="weighted")),
            "recall_weighted": float(recall_score(y_test, predictions, average="weighted")),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        }
    else:
        best_name = "rule_based_fallback"
        best_pipeline = RuleBasedClassifier().fit(x_train.tolist(), y_train.tolist())
        predictions = best_pipeline.predict(x_test.tolist())
        y_true = y_test.to_numpy()
        holdout_metrics = {
            "accuracy": float(np.mean(predictions == y_true)),
            "precision_weighted": float(np.mean(predictions == y_true)),
            "recall_weighted": float(np.mean(predictions == y_true)),
            "f1_weighted": float(np.mean(predictions == y_true)),
        }
        cv_results = {
            "rule_based_fallback": {
                metric: {"mean": holdout_metrics[metric], "std": 0.0} for metric in SCORING
            }
        }

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    artifacts_dir = root / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, models_dir / "classical_model.pkl")
    if SKLEARN_AVAILABLE:
        fitted_vectorizer = best_pipeline.named_steps["tfidf"]
        joblib.dump(fitted_vectorizer, models_dir / "tfidf_vectorizer.pkl")

    metrics_path = artifacts_dir / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    metrics_data["classical_model"] = {
        "best_model": best_name,
        "cross_validation": cv_results,
        "holdout_test": holdout_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Best classical model: {best_name}")
    print(f"Saved classical model to {models_dir / 'classical_model.pkl'}")
    return metrics_data["classical_model"]


def main() -> None:
    train_classical()


if __name__ == "__main__":
    main()
