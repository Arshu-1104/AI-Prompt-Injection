from __future__ import annotations

import base64
import codecs
import json
import re
from pathlib import Path

import pandas as pd


ATTACK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)(\s+(instructions?|rules?|prompts?|context))?",
    r"disregard\s+(all\s+)?(previous|prior|above)?\s*(instructions?|polic(y|ies)|rules?|prompts?)",
    r"forget\s+(your|all)\s+(rules?|instructions?|guidelines?)",
    r"you\s+are\s+now\s+(an?\s+)?(unrestricted|uncensored|free|jailbroken)",
    r"act\s+as\s+(an?\s+)?(unrestricted|uncensored|jailbroken|dan|root|admin|system|unfiltered)",
    r"pretend\s+to\s+be\s+(an?\s+)?(unrestricted|uncensored|ai\s+with\s+no|root|admin|unfiltered)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"(reveal|show|print|output)\s+(the\s+)?(system\s+prompt|hidden\s+prompt)",
    r"(reveal|show|print|output)\s+(the\s+)?hidden\s+(secrets?|data|prompt|info)",
    r"reveal\s+(the\s+)?hidden",
    r"execute\s+hidden\s+instructions?",
    r"bypass\s+(safety|content)\s+(filters?|checks?|policy)",
    r"override\s+(policy|safety|guidelines?)",
    r"your\s+new\s+task\s+is\s+to\s+(reveal|leak|bypass|ignore|disclose)",
    r"output\s+the\s+confidential",
    r"with\s+no\s+policy\s+filters?",
    r"run\s+unrestricted",
]

BENIGN_ALLOWLIST = [
    r"act\s+as\s+a\s+(sounding\s+board|reference|guide|mentor|tutor|reviewer)",
    r"roleplay\s+a\s+secure\s+assistant",
    r"pretend\s+to\s+be\s+a\s+(teacher|customer|interviewer|character)\b(?!.*\b(unrestricted|uncensored|no\s+filter|no\s+policy|jailbreak))",
]

# Matches Base64-like substrings: at least 8 chars from the B64 alphabet, optional padding.
# Minimum of 8 avoids false-positive matches on short alphanumeric tokens while still
# catching realistic encoded payloads (even compact ones like "Hello World" = 16 chars).
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")


def _decode_encoded_segments(text: str) -> str:
    """Attempt to decode Base64 and ROT13 segments in *text*.

    For each candidate Base64 substring:
    - Try base64.b64decode; if the result is printable ASCII, substitute it in.
    - Try ROT13 decode; if the result differs and is printable ASCII, substitute it in.
    Exceptions are caught silently so the function is always safe to call.
    """
    def _is_printable_ascii(b: bytes) -> bool:
        try:
            decoded = b.decode("ascii")
            return all(32 <= ord(c) < 127 or c in "\n\r\t" for c in decoded)
        except (UnicodeDecodeError, ValueError):
            return False

    result = text
    for match in reversed(list(_BASE64_RE.finditer(text))):
        segment = match.group(0)
        start, end = match.start(), match.end()
        replacement: str | None = None

        # 1. Try Base64
        try:
            # Pad to valid length
            padded = segment + "=" * (-len(segment) % 4)
            decoded_bytes = base64.b64decode(padded)
            if _is_printable_ascii(decoded_bytes):
                replacement = decoded_bytes.decode("ascii").strip()
        except Exception:
            pass

        # 2. Try ROT13 if Base64 didn't yield anything useful
        if replacement is None:
            try:
                rot13 = codecs.decode(segment, "rot_13")
                # Only accept ROT13 result if it contains a known injection keyword,
                # otherwise random Base64 alphabet strings produce meaningless garbage
                injection_keywords = {
                    "ignore", "bypass", "jailbreak", "reveal",
                    "forget", "disregard", "override", "system prompt",
                }
                if any(kw in rot13.lower() for kw in injection_keywords):
                    replacement = rot13
            except Exception:
                pass

        if replacement is not None:
            result = result[:start] + replacement + result[end:]

    return result


def clean_text(text: str) -> str:
    # Decode any Base64 / ROT13 payloads before further normalisation
    text = _decode_encoded_segments(str(text))
    cleaned = text.lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def highlight_attack_patterns(text: str, *, already_clean: bool = False) -> list[str]:
    lower = text if already_clean else clean_text(text)
    for benign_pattern in BENIGN_ALLOWLIST:
        if re.search(benign_pattern, lower, flags=re.IGNORECASE):
            return []
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
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    skipped_labels: list[str] = []
    for label, group in df.groupby("label", sort=False):
        if len(group) < 2:
            skipped_labels.append(str(label))
            train_parts.append(group)
            continue
        shuffled = group.sample(frac=1.0, random_state=random_state)
        split_idx = min(max(int(len(shuffled) * (1.0 - test_size)), 1), len(shuffled) - 1)
        train_parts.append(shuffled.iloc[:split_idx])
        test_parts.append(shuffled.iloc[split_idx:])

    if skipped_labels:
        print(
            f"[get_train_test_data] WARNING: {skipped_labels} had <2 samples — "
            "placed entirely in train. These classes will show 0 support in test metrics."
        )

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=random_state)
    test_df = (
        pd.concat(test_parts).sample(frac=1.0, random_state=random_state)
        if test_parts
        else pd.DataFrame(columns=df.columns)
    )

    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    split_path = artifacts_dir / "split_indices.json"
    split_payload = {
        "train_indices": train_df.index.tolist(),
        "test_indices": test_df.index.tolist(),
        "skipped_labels": skipped_labels,
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
