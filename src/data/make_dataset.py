from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def _to_categorical(series: np.ndarray, bins: int, labels: list[str]) -> list[str]:
    quantized = pd.qcut(series, q=bins, labels=labels, duplicates="drop")
    return quantized.astype(str).tolist()


def create_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a realistic synthetic employee attrition dataset.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        weights=[0.72, 0.28],
        class_sep=1.0,
        random_state=random_state,
    )

    df_num = pd.DataFrame(
        X,
        columns=[
            "performance_score",
            "monthly_hours",
            "commute_distance",
            "overtime_index",
            "salary_proxy",
            "engagement_proxy",
            "project_load",
            "tenure_proxy",
        ],
    )

    rng = np.random.default_rng(random_state)

    dataset = pd.DataFrame(
        {
            "age": np.clip((df_num["tenure_proxy"] * 6 + 35).round(), 21, 60).astype(int),
            "tenure_years": np.clip((df_num["tenure_proxy"] * 2 + 5).round(), 0, 30).astype(int),
            "monthly_hours": np.clip((df_num["monthly_hours"] * 20 + 170).round(), 80, 320).astype(int),
            "overtime_hours": np.clip((df_num["overtime_index"] * 12 + 10).round(), 0, 50).astype(int),
            "salary_usd": np.clip((df_num["salary_proxy"] * 18000 + 75000).round(), 25000, 180000).astype(int),
            "engagement_score": np.clip((df_num["engagement_proxy"] * 15 + 70).round(), 20, 100).astype(int),
            "distance_km": np.clip((df_num["commute_distance"] * 6 + 12).round(), 1, 45).astype(int),
            "projects_per_quarter": np.clip((df_num["project_load"] * 1.8 + 3).round(), 1, 8).astype(int),
            "department": rng.choice(
                ["Engineering", "Sales", "HR", "Operations", "Support"],
                size=n_samples,
                p=[0.33, 0.25, 0.09, 0.2, 0.13],
            ),
            "work_mode": rng.choice(
                ["Onsite", "Hybrid", "Remote"], size=n_samples, p=[0.4, 0.45, 0.15]
            ),
            "education_level": _to_categorical(
                df_num["performance_score"].to_numpy(),
                bins=3,
                labels=["Bachelors", "Masters", "PhD"],
            ),
            "satisfaction_band": _to_categorical(
                df_num["engagement_proxy"].to_numpy(),
                bins=3,
                labels=["Low", "Medium", "High"],
            ),
            "attrition": y.astype(int),
        }
    )
    return dataset


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    (root / "src" / "__init__.py").touch(exist_ok=True)
    (root / "src" / "data" / "__init__.py").touch(exist_ok=True)
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "employee_attrition.csv"

    dataset = create_dataset()
    dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to: {output_file}")
    print(dataset.head(3))


if __name__ == "__main__":
    main()
