"""A4 training helpers. Skipped where transformers/torch cannot be imported."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.transformer_training import NAN_CHECK_STEPS, TrainingError, build_arg_parser, token_length_report
from src.train_bert import SPEC

transformers = pytest.importorskip("transformers")
try:
    import torch  # noqa: F401
except Exception:  # torch blocked or missing
    pytest.skip("torch not importable", allow_module_level=True)

from src.transformer_training import _callbacks  # noqa: E402


def test_defaults():
    args = build_arg_parser(SPEC).parse_args([])
    assert args.max_length == 128 and args.batch_size is None and args.early_stopping_patience == 1
    args = build_arg_parser(SPEC).parse_args(["--resume", "--checkpoint-dir", "/drive/ckpt", "--keep-checkpoints"])
    assert args.resume and str(args.checkpoint_dir).endswith("ckpt") and args.keep_checkpoints


def test_nan_guard_aborts_early_but_not_later():
    guard, _ = _callbacks(transformers)
    with pytest.raises(TrainingError, match="Loss is nan"):
        guard.on_log(None, SimpleNamespace(global_step=10), None, logs={"loss": math.nan})
    guard.on_log(None, SimpleNamespace(global_step=NAN_CHECK_STEPS + 1), None, logs={"loss": math.nan})
    guard.on_log(None, SimpleNamespace(global_step=5), None, logs={"loss": 0.7})


def test_epoch_timer_records_each_epoch():
    _, timer = _callbacks(transformers)
    for _ in range(2):
        timer.on_epoch_begin(None, None, None)
        timer.on_epoch_end(None, None, None)
    assert len(timer.epoch_seconds) == 2 and all(t >= 0 for t in timer.epoch_seconds)


def test_token_length_report_counts_truncation():
    class Tok:
        def __call__(self, texts, truncation=False):
            return {"input_ids": [[0] * len(t.split()) for t in texts]}

    report = token_length_report(Tok(), ["a b", "a b c d e", "a"], max_length=3)
    assert report["max"] == 5 and report["truncated_pct"] == pytest.approx(100 / 3)
