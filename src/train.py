from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_and_evaluate(data_path: Path, artifacts_dir: Path, random_state: int = 42) -> None:
    df = pd.read_csv(data_path)
    target_col = "attrition"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1200, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4, random_state=random_state
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, dict[str, float]] = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None
    best_predictions = None
    best_probabilities = None
    model_probabilities: dict[str, pd.Series] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", clone(preprocess)), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        model_probabilities[model_name] = pd.Series(probs).reset_index(drop=True)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        comparison[model_name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = model_name
            best_pipeline = pipeline
            best_predictions = preds
            best_probabilities = probs

    if best_name is None or best_pipeline is None:
        raise ValueError("Best model selection failed: no valid trained pipeline was found.")
    if best_predictions is None or best_probabilities is None:
        raise ValueError("Best model predictions are missing after training.")

    y_test_series = y_test.reset_index(drop=True)

    outputs = {
        "project": "Employee Attrition Risk Prediction",
        "best_model": best_name,
        "model_comparison": comparison,
        "best_model_details": {
            "classification_report": classification_report(
                y_test_series, best_predictions, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test_series, best_predictions).tolist(),
        },
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    scored = X_test.reset_index(drop=True).copy()
    scored["actual_attrition"] = y_test_series
    scored["predicted_attrition"] = best_predictions
    scored["attrition_probability"] = best_probabilities
    for model_name, probs in model_probabilities.items():
        scored[f"{model_name}_probability"] = probs
    scored.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    joblib.dump(best_pipeline, model_dir / "best_model.joblib")
    print(f"Best model: {best_name}")
    print(f"Saved metrics to: {artifacts_dir / 'metrics.json'}")
    print(f"Saved model to: {model_dir / 'best_model.joblib'}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "employee_attrition.csv"
    artifacts_dir = root / "artifacts"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run src/data/make_dataset.py first."
        )

    train_and_evaluate(data_path=data_path, artifacts_dir=artifacts_dir)


if __name__ == "__main__":
    main()
