from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.preprocess import get_train_test_data


SCORING = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]


def train_classical(random_state: int = 42) -> dict:
    train_df, test_df = get_train_test_data(random_state=random_state)
    x_train = train_df["text"]
    y_train = train_df["label"]
    x_test = test_df["text"]
    y_test = test_df["label"]

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True)
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
        pipeline = Pipeline(
            steps=[("tfidf", clone(vectorizer)), ("classifier", clone(model))]
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

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    artifacts_dir = root / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, models_dir / "classical_model.pkl")
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
