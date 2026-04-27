from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    comparison = pd.DataFrame(metrics["model_comparison"]).T.reset_index()
    comparison.rename(columns={"index": "model"}, inplace=True)
    comparison.to_csv(root / "artifacts" / "model_comparison.csv", index=False)
    print(f"Saved figures and table to: {root / 'artifacts'}")


if __name__ == "__main__":
    main()
