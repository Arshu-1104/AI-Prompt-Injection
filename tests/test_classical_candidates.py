"""Classical candidate table: reporting only, consistent with the selection rule."""
from __future__ import annotations

import pytest

from src.train_classical import candidate_table


def _v(score: float, n: int = 1162) -> dict:
    return {"three_class": {"f1_weighted": score, "n_samples": n}}


def test_ranking_gap_and_rows():
    validation = {"a": _v(0.9590), "b": _v(0.960455), "c": _v(0.959559), "nb": _v(0.8867)}
    best = max(validation, key=lambda m: validation[m]["three_class"]["f1_weighted"])  # the selection rule
    table = candidate_table(validation, best)
    assert table["winner"] == best == table["ranking"][0]["config"] == "b"
    assert [r["config"] for r in table["ranking"]] == ["b", "c", "a", "nb"]
    assert table["gap_best_to_second"] == pytest.approx(0.960455 - 0.959559)
    assert table["gap_in_validation_rows"] == pytest.approx((0.960455 - 0.959559) * 1162)
    assert table["n_validation_rows"] == 1162


def test_exact_tie_keeps_the_selected_winner_first():
    validation = {"first": _v(0.95), "second": _v(0.95)}
    best = max(validation, key=lambda m: validation[m]["three_class"]["f1_weighted"])
    table = candidate_table(validation, best)
    assert table["ranking"][0]["config"] == best == "first"
    assert table["gap_best_to_second"] == 0


def test_single_candidate_has_no_gap():
    table = candidate_table({"rule_based_fallback": _v(0.5)}, "rule_based_fallback")
    assert table["gap_best_to_second"] is None and table["gap_in_validation_rows"] is None
