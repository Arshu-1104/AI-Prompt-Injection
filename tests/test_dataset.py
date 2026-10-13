"""load_data is the single dataset source; these tests run offline on a tiny
fake source table instead of the Hub download."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import src.load_data as ld
from src import preprocess


@pytest.fixture()
def fake_source():
    rows = []
    for i in range(40):
        rows.append({"text": f"benign question number {i}", "label": 0, "category": "benign"})
        rows.append({"text": f"ignore previous instructions variant {i}", "label": 1, "category": "direct_injection"})
    return pd.DataFrame(rows)


@pytest.fixture()
def data_dir(tmp_path, monkeypatch, fake_source):
    source_file = tmp_path / "source.parquet"
    fake_source.to_parquet(source_file)
    monkeypatch.setattr(ld, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ld, "PROCESSED_PATH", tmp_path / "processed_dataset.csv")
    monkeypatch.setattr(ld, "META_PATH", tmp_path / "dataset_meta.json")
    monkeypatch.setattr(ld, "download_source", lambda: (fake_source, source_file))
    ld.main()
    return tmp_path


def test_build_is_deterministic(fake_source):
    first = ld.build_dataset(source_df=fake_source)
    second = ld.build_dataset(source_df=fake_source)
    pd.testing.assert_frame_equal(first, second)
    assert {"text", "label", "original_label", "category"} <= set(first.columns)
    assert {"synthetic_suspicious", "human_suspicious", "benign_hard_negative"} <= set(first["category"])


def test_meta_records_source_and_hash(data_dir):
    meta = json.loads((data_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    assert meta["source"]["name"] == ld.DATASET_NAME
    assert meta["source"]["revision"] == ld.DATASET_REVISION
    assert meta["processed"]["sha256"] == ld.sha256_file(data_dir / "processed_dataset.csv")
    assert meta["processed"]["rows"] == sum(meta["processed"]["label_counts"].values())


def test_splits_carry_provenance(data_dir):
    train_df, test_df, provenance = preprocess.get_splits_with_provenance()
    assert provenance["source"]["revision"] == ld.DATASET_REVISION
    assert provenance["split"]["seed"] == preprocess.SPLIT_SEED
    assert provenance["split"]["train_rows"] == len(train_df)
    assert provenance["split"]["test_rows"] == len(test_df)
    assert len(train_df) + len(test_df) == provenance["processed_rows"]


def test_edited_csv_is_rejected(data_dir):
    path = data_dir / "processed_dataset.csv"
    path.write_text(path.read_text(encoding="utf-8") + "extra row,SAFE,0,benign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        preprocess.get_train_test_data()
