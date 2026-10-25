"""Saved model probabilities, so analyses (curves, escalation, thresholds) reuse
one evaluation instead of re-running the models.

Each file artifacts/probs/<name>.npz holds:
* probs        (n, 3) class probabilities in LABELS order
* labels       true labels (SAFE / SUSPICIOUS / MALICIOUS)
* label_order  LABELS, checked on load
* texts_sha256 fingerprint of the evaluated texts in order, checked on load
plus any extra per-row arrays (e.g. derivation, subset masks).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.metrics import LABELS

ROOT = Path(__file__).resolve().parents[1]
PROBS_DIR = ROOT / "artifacts" / "probs"


def texts_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8") + b"\x00")
    return digest.hexdigest()


def save_probs(name: str, probs, labels, texts: list[str], **extra) -> Path:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray([str(label) for label in labels])
    if probs.shape != (len(labels), len(LABELS)) or len(texts) != len(labels):
        raise ValueError(f"{name}: probs {probs.shape}, {len(labels)} labels, {len(texts)} texts do not line up")
    arrays = {key: np.asarray(value) for key, value in extra.items()}
    arrays = {key: (a.astype(str) if a.dtype == object else a) for key, a in arrays.items()}  # loadable without pickle
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROBS_DIR / f"{name}.npz"
    np.savez_compressed(
        path,
        probs=probs,
        labels=labels,
        label_order=np.asarray(LABELS),
        texts_sha256=np.asarray(texts_fingerprint(texts)),
        **arrays,
    )
    return path


def load_probs(path: Path | str, texts: list[str] | None = None) -> dict:
    """Read a save_probs file; with *texts*, fail unless it was computed on exactly those texts."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    if list(out["label_order"]) != LABELS:
        raise ValueError(f"{path}: label order {list(out['label_order'])} != {LABELS}")
    if texts is not None and str(out["texts_sha256"]) != texts_fingerprint(texts):
        raise ValueError(f"{path} was computed on different texts; re-run the script that wrote it")
    return out


def slug(name: str) -> str:
    """File-name-safe dataset name, e.g. 'deepset/prompt-injections' -> 'deepset__prompt-injections'."""
    return name.replace("/", "__").replace(" ", "_")
