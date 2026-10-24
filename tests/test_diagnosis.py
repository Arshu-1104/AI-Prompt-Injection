from __future__ import annotations

from src.diagnosis import sentence_form, shares


def test_sentence_form():
    assert sentence_form("Ignore all previous instructions.") == "imperative"
    assert sentence_form("Please translate this into French") == "imperative"
    assert sentence_form("Given two words, output the longer one.") == "imperative"  # verb starts the second clause
    assert sentence_form("The weather is nice today.") == "other"
    assert sentence_form("What is the capital of France?") == "question"
    assert sentence_form("Tell me a joke?") == "question"  # trailing '?' wins
    assert sentence_form("I like trains") == "other"
    assert sentence_form("") == "other"


def test_shares_sum_to_one_with_intervals():
    s = shares(["Write a poem", "Why?", "hello there", "List three fruits"])
    assert s["n"] == 4
    assert abs(sum(s[f]["share"] for f in ("imperative", "question", "other")) - 1) < 1e-9
    assert s["imperative"]["share"] == 0.5
    lo, hi = s["imperative"]["wilson95"]
    assert lo < 0.5 < hi
