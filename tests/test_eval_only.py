"""eval_only must work for the classical model with no torch and no GPU."""
from __future__ import annotations

import pytest

import src.load_data as ld


def test_eval_only_classical_cpu(monkeypatch):
    if not all(p.exists() for p in ld.SPLIT_PATHS.values()):
        pytest.skip("data/*.csv not built; run `python -m src.load_data`")
    import src.evaluate as evaluate
    from src import eval_only

    written = {}
    monkeypatch.setattr(evaluate, "_write_artifacts", lambda rows, outputs, df, a, f, prov: written.update(rows=rows, prov=prov))
    rows = eval_only.main(["--models", "classical"])
    assert rows[0]["model"] == "classical" and rows[0]["status"] == "ok"
    assert {"binary", "three_class", "pipeline_with_rules"} <= set(rows[0])
    assert written["prov"]["evaluated_with"].startswith("src.eval_only")


def test_eval_only_does_not_import_training_code_at_module_level():
    import importlib
    import sys

    sys.modules.pop("src.eval_only", None)
    sys.modules.pop("src.transformer_training", None)
    importlib.import_module("src.eval_only")
    assert "src.transformer_training" not in sys.modules
