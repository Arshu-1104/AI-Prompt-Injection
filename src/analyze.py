from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
