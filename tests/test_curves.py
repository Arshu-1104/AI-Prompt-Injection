from __future__ import annotations

import numpy as np
import pytest

from src import curves, probs_io


def _probs(p_safe: list[float]) -> np.ndarray:
    """3-class rows with the given P(SAFE); the rest goes to MALICIOUS."""
    p = np.asarray(p_safe, dtype=float)
    return np.column_stack([p, np.zeros_like(p), 1 - p])


def test_probs_round_trip_and_fingerprint_check(tmp_path, monkeypatch):
    monkeypatch.setattr(probs_io, "PROBS_DIR", tmp_path)
    texts = ["a", "b"]
    import pandas as pd

    derivation = pd.Series(["original"] * 2).astype(object)  # pandas object column: must still load without pickle
    path = probs_io.save_probs("x", _probs([0.9, 0.2]), ["SAFE", "MALICIOUS"], texts, derivation=derivation)
    saved = probs_io.load_probs(path, texts)
    assert saved["probs"].shape == (2, 3) and list(saved["labels"]) == ["SAFE", "MALICIOUS"]
    assert list(saved["derivation"]) == ["original", "original"]
    with pytest.raises(ValueError, match="different texts"):
        probs_io.load_probs(path, ["a", "c"])
    with pytest.raises(ValueError, match="do not line up"):
        probs_io.save_probs("bad", _probs([0.5]), ["SAFE", "SAFE"], texts)


def test_fpr_at_tpr_picks_the_highest_threshold_reaching_the_target():
    # attack scores: positives 0.9, 0.8, 0.3; negatives 0.7, 0.2, 0.1
    y = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    score = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    fpr, threshold = curves.fpr_at_tpr(y, score, 0.60)  # 2 of 3 positives: threshold 0.8, no negative above it
    assert (fpr, threshold) == (0.0, 0.8)
    fpr, threshold = curves.fpr_at_tpr(y, score, 0.90)  # all positives: threshold 0.3 lets the 0.7 negative in
    assert (round(fpr, 6), threshold) == (round(1 / 3, 6), 0.3)


def test_confusion_counts_use_the_api_rule():
    y = np.array([True, True, False, False])
    probs = np.array([[0.2, 0.7, 0.1], [0.6, 0.0, 0.4], [0.3, 0.6, 0.1], [0.9, 0.0, 0.1]])  # SUSPICIOUS counts as attack
    c = curves.confusion_binary(y, probs)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)


def test_confusion_3class_rows_are_true_labels():
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    assert curves.confusion_3class(["SAFE", "SAFE", "MALICIOUS"], probs) == [[1, 0, 1], [0, 0, 0], [0, 1, 0]]


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.random(300) < 0.5
    score = np.where(y, rng.normal(0.7, 0.2, 300), rng.normal(0.4, 0.2, 300))
    point = curves.point_metrics(y, score)
    ci = curves.bootstrap(y, score, n_boot=200)
    for key in ("auroc", "auprc", "fpr_at_tpr90", "fpr_at_tpr95"):
        assert ci[key][0] <= point[key] <= ci[key][1], key


def test_external_runs_parses_dataset_and_model_names(tmp_path, monkeypatch):
    monkeypatch.setattr(probs_io, "PROBS_DIR", tmp_path)
    for name in ("external_deepset__prompt-injections_classical", "external_deepset__prompt-injections_llama_pg2",
                 "external_qualifire__prompt-injections-benchmark_guard", "test_classical_seed42"):
        (tmp_path / f"{name}.npz").write_bytes(b"")
    runs = curves.external_runs()
    assert set(runs) == {"deepset__prompt-injections", "qualifire__prompt-injections-benchmark"}
    assert set(runs["deepset__prompt-injections"]) == {"classical", "llama_pg2"}
    assert curves.unslug("deepset__prompt-injections") == "deepset/prompt-injections"
