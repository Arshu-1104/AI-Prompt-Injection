"""Train/validation/test leakage checks.

Unit tests build tiny fake splits with planted collisions. The data tests run on
the real data/{train,validation,test}.csv when present (CI builds them with
`python -m src.load_data` before pytest) and are skipped otherwise.
"""
from __future__ import annotations

import pandas as pd
import pytest

import src.load_data as ld

# --------------------------------------------------------------------------- planted collisions


def _rows(split, items):
    return pd.DataFrame(
        [
            {"text": text, "label": label, "category": "x", "source": "s", "group_id": group}
            for text, label, group in items
        ]
    )


def _base():
    train = _rows("train", [(f"train benign {i}", 0, f"tb{i}") for i in range(30)]
                  + [(f"train attack {i}", 1, f"tm{i}") for i in range(30)])
    val = _rows("validation", [(f"val benign {i}", 0, f"vb{i}") for i in range(10)]
                + [(f"val attack {i}", 1, f"vm{i}") for i in range(10)])
    test = _rows("test", [(f"test benign {i}", 0, f"sb{i}") for i in range(10)]
                 + [(f"test attack {i}", 1, f"sm{i}") for i in range(10)])
    return {"train": train, "validation": val, "test": test}


def _build(sources):
    return ld.remove_cross_split_overlap({s: ld.build_split(sources[s], s) for s in ld.SPLITS})


def test_train_copy_of_a_test_row_is_dropped_from_train():
    sources = _base()
    # Same prompt in train (different group, different spacing/case) and in test.
    sources["train"].loc[len(sources["train"])] = ["TEST  attack 3", 1, "x", "s", "tm_dup"]
    built, report = _build(sources)
    train_keys = set(built["train"]["text"].map(ld.dedup_key))
    assert ld.dedup_key("test attack 3") not in train_keys
    assert report["train"]["rows_dropped"] >= 1
    assert "test attack 3" in set(built["test"]["text"])  # official test rows are kept


def test_derived_eval_row_colliding_with_train_is_dropped():
    sources = _base()
    leet_of_val = "val attack 0".translate(ld.LEET)
    sources["train"].loc[len(sources["train"])] = [leet_of_val, 1, "x", "s", "tm_leet"]
    built, report = _build(sources)
    for split in ("validation", "test"):
        derived = built[split][built[split]["derivation"] != "original"]
        assert not set(derived["text"].map(ld.dedup_key)) & set(built["train"]["text"].map(ld.dedup_key))


def test_shared_group_between_splits_fails():
    sources = _base()
    sources["test"].loc[0, "group_id"] = "tb0"
    with pytest.raises(RuntimeError, match="group_ids"):
        _build(sources)


def test_hand_authored_samples_land_in_exactly_one_split():
    for text in ld.HUMAN_SUSPICIOUS_SAMPLES + ld.BENIGN_HARD_NEGATIVES:
        assert ld.hand_authored_split(text) in ld.SPLITS
    built, _ = _build(_base())
    seen: dict[str, list[str]] = {}
    for split, df in built.items():
        for text in df.loc[df["derivation"].isin(["human_suspicious", "hard_negative"]), "text"]:
            seen.setdefault(text, []).append(split)
    assert all(len(splits) == 1 for splits in seen.values())


# --------------------------------------------------------------------------- real data


def _real_splits():
    if not all(path.exists() for path in ld.SPLIT_PATHS.values()):
        pytest.skip("data/*.csv not built; run `python -m src.load_data`")
    return {s: pd.read_csv(ld.SPLIT_PATHS[s], keep_default_na=False) for s in ld.SPLITS}


def test_real_data_no_text_overlap_across_splits():
    splits = _real_splits()
    keys = {s: set(df["text"].map(ld.dedup_key)) - {""} for s, df in splits.items()}
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["test"]
    assert not keys["validation"] & keys["test"]


def test_real_data_no_group_overlap_across_splits():
    splits = _real_splits()
    groups = {s: set(df["group_id"]) for s, df in splits.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]


def test_real_data_variants_stay_with_their_source_row():
    for split, df in _real_splits().items():
        originals = set(df.loc[df["derivation"] == "original", "group_id"])
        variants = df[df["derivation"].isin(["synthetic_suspicious", "leet"])]
        assert set(variants["group_id"]) <= originals, split


def test_robustness_prompts_and_training_extras_do_not_leak():
    from src.robustness import ADVERSARIAL_PROMPTS
    from src.train_classical import EVERYDAY_SAFE_EXAMPLES

    splits = _real_splits()
    keys = {s: set(df["text"].map(ld.dedup_key)) for s, df in splits.items()}
    adversarial = {ld.dedup_key(item["text"]) for item in ADVERSARIAL_PROMPTS}
    assert not adversarial & keys["train"], "robustness prompts must not be training data"
    everyday = {ld.dedup_key(t) for t in EVERYDAY_SAFE_EXAMPLES}
    assert not everyday & (keys["validation"] | keys["test"]), "train-only extras must not be eval data"
