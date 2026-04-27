from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import auc, roc_curve

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-specific fallback
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "employee_attrition.csv"
    metrics_path = root / "artifacts" / "metrics.json"
    model_path = root / "artifacts" / "models" / "best_model.joblib"
    predictions_path = root / "artifacts" / "test_predictions.csv"
    figures_dir = root / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "Missing dataset or metrics. Run make_dataset.py and train.py first."
        )

    df = pd.read_csv(data_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    comparison = pd.DataFrame(metrics["model_comparison"]).T.reset_index()
    comparison.rename(columns={"index": "model"}, inplace=True)
    comparison.to_csv(root / "artifacts" / "model_comparison.csv", index=False)
    comparison_path = root / "artifacts" / "model_comparison.csv"

    summary = {
        "rows": int(len(df)),
        "attrition_rate": float(df["attrition"].mean()),
        "avg_overtime_hours_attrition_1": float(df.loc[df["attrition"] == 1, "overtime_hours"].mean()),
        "avg_overtime_hours_attrition_0": float(df.loc[df["attrition"] == 0, "overtime_hours"].mean()),
    }
    with open(root / "artifacts" / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x="attrition")
        plt.title("Class Distribution: Attrition")
        plt.xlabel("Attrition (0=No, 1=Yes)")
        plt.tight_layout()
        plt.savefig(figures_dir / "class_distribution.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x="attrition", y="overtime_hours")
        plt.title("Overtime Hours vs Attrition")
        plt.xlabel("Attrition")
        plt.tight_layout()
        plt.savefig(figures_dir / "overtime_vs_attrition.png", dpi=140)
        plt.close()

        if not model_path.exists() or not predictions_path.exists() or not comparison_path.exists():
            raise FileNotFoundError(
                "Missing model or prediction artifacts. Run src/train.py before src/analyze.py."
            )

        best_pipeline = joblib.load(model_path)
        preprocess = best_pipeline.named_steps["preprocess"]
        model = best_pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Best model does not provide feature importances.")

        feature_names = preprocess.get_feature_names_out()
        feature_importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(9, 6))
        sns.barplot(
            data=feature_importance_df,
            y="feature",
            x="importance",
            orient="h",
            hue="feature",
            palette="viridis",
            legend=False,
        )
        plt.title("Top 15 Feature Importances (Best Model)")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png", dpi=140)
        plt.close()

        predictions_df = pd.read_csv(predictions_path)
        model_comparison_df = pd.read_csv(comparison_path)
        y_true = predictions_df["actual_attrition"]

        plt.figure(figsize=(12, 5))
        for idx, model_name in enumerate(model_comparison_df["model"], start=1):
            prob_col = f"{model_name}_probability"
            if prob_col not in predictions_df.columns:
                raise ValueError(
                    f"Missing '{prob_col}' in test predictions. Re-run src/train.py to regenerate predictions."
                )

            fpr, tpr, _ = roc_curve(y_true, predictions_df[prob_col])
            roc_auc = auc(fpr, tpr)

            plt.subplot(1, 2, idx)
            plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            plt.title(f"ROC Curve: {model_name}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(figures_dir / "roc_curve.png", dpi=140)
        plt.close()
        print(f"Saved figures and tables to: {root / 'artifacts'}")
    else:
        fallback_report = root / "artifacts" / "analysis_report.md"
        fallback_report.write_text(
            "\n".join(
                [
                    "# Analysis Report (Fallback Mode)",
                    "",
                    "Matplotlib/Seaborn plots could not be generated in this environment.",
                    f"- Import error: `{PLOTTING_IMPORT_ERROR}`",
                    "",
                    "## Generated outputs",
                    "- `artifacts/model_comparison.csv`",
                    "- `artifacts/analysis_summary.json`",
                    "- `artifacts/figures/feature_importance.png` (if plotting available)",
                    "- `artifacts/figures/roc_curve.png` (if plotting available)",
                    "",
                    "## Quick findings",
                    f"- Rows: {summary['rows']}",
                    f"- Attrition rate: {summary['attrition_rate']:.2%}",
                    f"- Avg overtime (attrition=1): {summary['avg_overtime_hours_attrition_1']:.2f}",
                    f"- Avg overtime (attrition=0): {summary['avg_overtime_hours_attrition_0']:.2f}",
                ]
            ),
            encoding="utf-8",
        )
        print("Plotting libraries unavailable due to system policy.")
        print(f"Saved fallback analysis report to: {fallback_report}")
        print(f"Saved tables to: {root / 'artifacts'}")


if __name__ == "__main__":
    main()
