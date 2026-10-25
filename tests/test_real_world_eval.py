from __future__ import annotations

import numpy as np
import pandas as pd

from src import real_world_eval as rw

SAFE, MAL = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]


def test_first_user_turn_skips_non_user_turns():
    conv = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi there", "toxic": True},
            {"role": "assistant", "content": "hello"}]
    assert rw.first_user_turn(conv) == ("hi there", True)
    assert rw.first_user_turn([{"role": "assistant", "content": "x"}]) == ("", False)


def test_dedupe_drops_empty_and_duplicate_texts():
    frame = pd.DataFrame({"text": ["a", "a", " ", "b"], "label": ["SAFE"] * 4})
    assert rw._dedupe(frame)["text"].tolist() == ["a", "b"]


def test_score_in_the_wild_reports_both_classes_and_sources():
    frame = pd.DataFrame({"text": [f"t{i}" for i in range(70)],
                          "label": ["MALICIOUS"] * 10 + ["SAFE"] * 60,
                          "source": ["x"] * 10 + ["siteA"] * 40 + ["siteB"] * 20})
    probs = np.array([MAL] * 10 + [MAL] * 4 + [SAFE] * 36 + [SAFE] * 20)
    out = rw.score_in_the_wild(frame, probs, n_boot=50)
    assert out["jailbreaks_detected"]["rate"] == 1.0
    assert out["regular_flagged"]["rate"] == 4 / 60
    assert set(out["regular_flagged_by_source"]) == {"siteA"}  # siteB has fewer than MIN_SOURCE_ROWS rows
    assert out["regular_flagged_by_source"]["siteA"]["rate"] == 4 / 40


def test_score_wildchat_flag_rates():
    frame = pd.DataFrame({"text": ["a", "b", "c", "d"], "toxic": [True, False, False, False], "label": ["UNLABELED"] * 4})
    out = rw.score_wildchat(frame, np.array([MAL, MAL, SAFE, SAFE]))
    assert out["flag_rate"]["rate"] == 0.5
    assert out["flag_rate_non_toxic"]["rate"] == 1 / 3
