from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ATTACK_PATTERNS = [
    r"ignore\s+previous",
    r"disregard\s+(all\s+)?instructions?",
    r"forget\s+(your|all)\s+rules?",
    r"you\s+are\s+now",
    r"act\s+as",
    r"pretend\s+to\s+be",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"system\s+prompt",
    r"reveal\s+(the\s+)?hidden",
    r"bypass\s+safety",
    r"override\s+policy",
    r"new\s+task\s+is",
    r"output\s+the\s+confidential",
]


def clean_text(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def highlight_attack_patterns(text: str) -> list[str]:
    lower = clean_text(text)
    matches: list[str] = []
    for pattern in ATTACK_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            matches.append(pattern)
    return matches


def get_risk_phrases(text: str) -> list[dict[str, str | int]]:
    spans: list[dict[str, str | int]] = []
    for pattern in ATTACK_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append(
                {
                    "pattern": pattern,
                    "start": int(match.start()),
                    "end": int(match.end()),
                    "text": match.group(0),
                }
            )
    spans.sort(key=lambda item: int(item["start"]))
    return spans


def load_processed_dataset() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run src/load_data.py first.")
    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("processed_dataset.csv must contain 'text' and 'label' columns.")
    df["text"] = df["text"].map(clean_text)
    return df


def get_train_test_data(test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_processed_dataset()
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    split_path = artifacts_dir / "split_indices.json"
    split_payload = {
        "train_indices": train_df.index.tolist(),
        "test_indices": test_df.index.tolist(),
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
