"""load_data is the single dataset source; these tests run offline on small
fake Hub splits instead of downloading."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import src.load_data as ld
from src import preprocess


def _fake_split(split: str, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"text": f"{split} benign question number {i}", "label": 0, "category": "benign",
                     "source": "original", "group_id": f"{split}_b{i}"})
        rows.append({"text": f"{split} ignore previous instructions variant {i}", "label": 1,
                     "category": "direct_injection", "source": "hackaprompt", "group_id": f"{split}_m{i}"})
    return pd.DataFrame(rows)


@pytest.fixture()
def fake_sources():
    return {"train": _fake_split("train", 60), "validation": _fake_split("validation", 20), "test": _fake_split("test", 20)}


@pytest.fixture()
def data_dir(tmp_path, monkeypatch, fake_sources):
    monkeypatch.setattr(ld, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ld, "META_PATH", tmp_path / "dataset_meta.json")
    monkeypatch.setattr(ld, "SPLIT_PATHS", {s: tmp_path / f"{s}.csv" for s in ld.SPLITS})
    monkeypatch.setattr(ld, "load_official_splits", lambda train_config="full": fake_sources)
    ld.main([])
    return tmp_path


def test_build_is_deterministic(fake_sources):
    first = ld.build_dataset(sources=fake_sources)
    second = ld.build_dataset(sources=fake_sources)
    for split in ld.SPLITS:
        pd.testing.assert_frame_equal(first[split], second[split])
    train = first["train"]
    assert {"text", "label", "hub_label", "derivation", "category", "source", "group_id"} <= set(train.columns)
    assert {"original", "synthetic_suspicious", "leet"} <= set(train["derivation"])


def test_official_rows_are_kept_and_not_resplit(fake_sources):
    built = ld.build_dataset(sources=fake_sources)
    for split in ld.SPLITS:
        originals = built[split][built[split]["derivation"] == "original"]
        assert sorted(originals["text"]) == sorted(fake_sources[split]["text"].str.strip())


def test_synthetic_suspicious_only_from_same_split(fake_sources):
    built = ld.build_dataset(sources=fake_sources)
    for split in ld.SPLITS:
        df = built[split]
        derived_groups = set(df.loc[df["derivation"].isin(["synthetic_suspicious", "leet"]), "group_id"])
        assert derived_groups <= set(df.loc[df["derivation"] == "original", "group_id"])
        assert df.loc[df["derivation"] != "original", "hub_label"].isna().all()


def test_size_check_fails_loudly(monkeypatch):
    class FakeDataset(dict):
        pass

    import datasets

    fake = FakeDataset(train=None)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)

    class Split:
        def __init__(self, n):
            self.n = n

        def to_pandas(self):
            return pd.DataFrame({"text": ["x"] * self.n})

    fake.update(train=Split(10), validation=Split(941), test=Split(942))
    with pytest.raises(RuntimeError, match="differ from the dataset card"):
        ld.load_official_splits("full")


def test_meta_records_source_license_and_hashes(data_dir):
    meta = json.loads((data_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    assert meta["dataset"]["name"] == ld.DATASET_NAME
    assert meta["dataset"]["revision"] == ld.DATASET_REVISION
    assert meta["dataset"]["train_config"] == "full"
    assert "Research" in meta["dataset"]["license"]["caveat"]
    for split in ld.SPLITS:
        assert meta["processed"][split]["sha256"] == ld.sha256_file(data_dir / f"{split}.csv")
    assert "synthetic" in meta["synthetic_note"].lower()


def test_splits_carry_provenance(data_dir):
    train_df, val_df, test_df, provenance = preprocess.get_splits()
    assert provenance["revision"] == ld.DATASET_REVISION
    assert provenance["processed"]["train"]["rows"] == len(train_df)
    assert provenance["processed"]["validation"]["rows"] == len(val_df)
    assert provenance["processed"]["test"]["rows"] == len(test_df)
    assert "license" in provenance and "synthetic_note" in provenance


def test_edited_csv_is_rejected(data_dir):
    path = data_dir / "test.csv"
    path.write_text(path.read_text(encoding="utf-8") + "extra row,SAFE,,original,benign,x,g\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        preprocess.load_split("test")
