from __future__ import annotations

import json

from src import paper_numbers as pn


def test_flatten_keys_lists_by_name_or_index_and_skips_text():
    out: dict = {}
    pn.flatten({"datasets": [{"name": "deepset/prompt-injections", "acc": 0.5}],
                "ci95": [0.1, 0.9], "flag": True, "label": "SAFE", "samples": [1, 2], "nan": float("nan"),
                "unsafe": "50% & $x$"}, "x", out)
    assert out == {"x.n_keys": 7, "x.datasets.deepset/prompt-injections.n_keys": 2,
                   "x.datasets.deepset/prompt-injections.acc": 0.5,
                   "x.datasets.deepset/prompt-injections.name": "deepset/prompt-injections",
                   "x.ci95.0": 0.1, "x.ci95.1": 0.9, "x.flag": 1, "x.label": "SAFE"}


def test_sanitize_keeps_keys_latex_safe():
    assert pn.sanitize("neuralchemy test (original rows)") == "neuralchemy_test__original_rows"
    assert pn.sanitize("bipia+override") == "bipia+override"


def test_main_writes_one_pgdef_per_number_with_its_source(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "curves.json").write_text(json.dumps({"sets": {"a": {"auroc": 0.75}}}), encoding="utf-8")
    monkeypatch.setattr(pn, "OUTPUT", tmp_path / "paper" / "numbers.tex")
    monkeypatch.setattr(pn, "ROOT", tmp_path)
    text = pn.main().read_text(encoding="utf-8")
    assert "\\pgdef{cur.sets.a.auroc}{0.75}% artifacts/curves.json" in text
    assert "paper_results.json" in text.splitlines()[2]  # listed as missing
