from __future__ import annotations

import numpy as np
import pytest

from src import stats
from src.metrics import LABELS, compute_ece, summarize_binary, summarize_predictions


@pytest.fixture()
def evaluated():
    rng = np.random.default_rng(1)
    n = 300
    labels = list(rng.choice(LABELS, size=n, p=[0.4, 0.15, 0.45]))
    derivation = ["original" if lab != "SUSPICIOUS" else "synthetic_suspicious" for lab in labels]
    probs = rng.dirichlet([1, 1, 1], size=n)
    for i, lab in enumerate(labels):  # make the model mostly right
        if rng.random() < 0.8:
            probs[i, LABELS.index(lab)] += 1.5
    probs = probs / probs.sum(axis=1, keepdims=True)
    return labels, derivation, probs


def test_vectorized_metrics_match_reference(evaluated):
    labels, derivation, probs = evaluated
    ev = stats.Evaluated(labels, derivation, probs)
    fast = ev.metrics()
    preds = [LABELS[i] for i in probs.argmax(axis=1)]
    ref = summarize_predictions(labels, preds, probs.max(axis=1).tolist())
    assert fast["three_class"]["accuracy"] == pytest.approx(ref["accuracy"])
    assert fast["three_class"]["f1_weighted"] == pytest.approx(ref["f1_weighted"])
    assert fast["three_class"]["ece"] == pytest.approx(ref["ece"])
    for label in LABELS:
        assert fast["three_class"][f"f1_{label}"] == pytest.approx(ref["per_class_f1"][label])
    original = np.array(derivation) == "original"
    ref_bin = summarize_binary([l for l, o in zip(labels, original) if o], probs[original])
    for metric in ("accuracy", "f1", "false_positive_rate", "ece"):
        assert fast["binary"][metric] == pytest.approx(ref_bin[metric])


def test_ece_bin_edges_match_reference():
    conf = np.array([0.3, 0.7, 1.0, 0.0, 0.1, 0.55])
    correct = np.array([1, 0, 1, 0, 1, 1], dtype=float)
    y = ["A"] * 6
    p = ["A" if c else "B" for c in correct]
    assert stats._ece(correct, conf) == pytest.approx(compute_ece(y, p, conf.tolist())[0])


def test_bootstrap_ci_contains_point_and_is_reproducible(evaluated):
    labels, derivation, probs = evaluated
    ev = stats.Evaluated(labels, derivation, probs)
    idx3, idxb = stats.resample_indices(len(labels), int(ev.original.sum()), n_boot=200, seed=0)
    ci = stats.bootstrap_ci(ev, idx3, idxb)
    for view in ("three_class", "binary"):
        entry = ci[view]["accuracy"]
        assert entry["ci95"][0] <= entry["point"] <= entry["ci95"][1]
    again = stats.bootstrap_ci(ev, *stats.resample_indices(len(labels), int(ev.original.sum()), n_boot=200, seed=0))
    assert again == ci


def test_mcnemar_exact_known_values():
    a = np.array([True] * 10 + [False] * 0 + [True] * 5)
    b = np.array([False] * 10 + [True] * 5)
    result = stats.mcnemar_exact(a, b)  # 10 vs 0 discordant -> p = 2 * 0.5**10
    assert (result["a_only_correct"], result["b_only_correct"]) == (10, 0)
    assert result["p_value"] == pytest.approx(2 * 0.5 ** 10)
    assert stats.mcnemar_exact(a, a)["p_value"] == 1.0


def test_paired_comparison_of_identical_models(evaluated):
    labels, derivation, probs = evaluated
    ev = stats.Evaluated(labels, derivation, probs)
    idx3, idxb = stats.resample_indices(len(labels), int(ev.original.sum()), n_boot=100)
    out = stats.paired_comparison(ev, ev, idx3, idxb)
    assert out["binary"]["difference"] == 0.0
    assert out["binary"]["paired_bootstrap_ci95"] == [0.0, 0.0]
    assert out["three_class"]["mcnemar"]["p_value"] == 1.0
