"""Run facts written into each bert/guard metrics.json at training time."""
from __future__ import annotations

from datetime import datetime

from src.transformer_training import _now_iso, checkpoint_step, training_provenance

FIELDS = {
    "training_started_at",
    "training_finished_at",
    "total_optimizer_steps",
    "steps_per_epoch",
    "best_epoch",
    "best_checkpoint_name",
    "resumed",
    "resumed_from_step",
}


def _history(steps_per_epoch: int, f1s: list[float]) -> list[dict]:
    history = [{"loss": 0.5, "step": 10, "epoch": 0.01}]
    for i, f1 in enumerate(f1s, start=1):
        history.append({"eval_f1": f1, "eval_loss": 0.2, "step": i * steps_per_epoch, "epoch": float(i)})
    return history


def _assert_well_formed(facts: dict) -> None:
    assert set(facts) == FIELDS
    for key in ("training_started_at", "training_finished_at"):
        stamp = datetime.fromisoformat(facts[key])
        assert stamp.tzinfo is not None, f"{key} has no timezone"
    assert datetime.fromisoformat(facts["training_started_at"]) <= datetime.fromisoformat(facts["training_finished_at"])
    assert isinstance(facts["total_optimizer_steps"], int) and facts["total_optimizer_steps"] >= 0
    assert isinstance(facts["steps_per_epoch"], int) and facts["steps_per_epoch"] > 0
    assert isinstance(facts["resumed"], bool)
    assert facts["resumed"] == (facts["resumed_from_step"] is not None)
    if facts["best_checkpoint_name"] is not None:
        assert facts["best_checkpoint_name"].startswith("checkpoint-")
        assert isinstance(facts["best_epoch"], int) and facts["best_epoch"] >= 1


def test_fresh_run_records_best_epoch_and_checkpoint_name():
    facts = training_provenance(
        started_at="2026-10-13T17:53:35+05:30",
        finished_at="2026-10-13T18:17:56+05:30",
        global_step=6339,
        steps_per_epoch=2113,
        best_model_checkpoint="artifacts/paper_runs/guard/seed_44_checkpoints/checkpoint-4226",
        log_history=_history(2113, [0.95785, 0.95787, 0.95615]),
        resumed_from_step=None,
    )
    _assert_well_formed(facts)
    assert facts["best_checkpoint_name"] == "checkpoint-4226"
    assert facts["best_epoch"] == 2
    assert facts["resumed"] is False and facts["resumed_from_step"] is None


def test_resumed_run_records_the_resume_step():
    facts = training_provenance(
        started_at=_now_iso(),
        finished_at=_now_iso(),
        global_step=4226,
        steps_per_epoch=2113,
        best_model_checkpoint="runs/seed_42_checkpoints/checkpoint-2113",
        log_history=_history(2113, [0.9648, 0.9587]),
        resumed_from_step=2113,
    )
    _assert_well_formed(facts)
    assert facts["resumed"] is True and facts["resumed_from_step"] == 2113
    assert (facts["best_epoch"], facts["best_checkpoint_name"]) == (1, "checkpoint-2113")


def test_smoke_run_without_checkpoints():
    facts = training_provenance(
        started_at=_now_iso(),
        finished_at=_now_iso(),
        global_step=4,
        steps_per_epoch=4,
        best_model_checkpoint=None,
        log_history=[],
        resumed_from_step=None,
    )
    _assert_well_formed(facts)
    assert facts["best_epoch"] is None and facts["best_checkpoint_name"] is None


def test_checkpoint_step():
    assert checkpoint_step("checkpoint-1587") == 1587
    assert checkpoint_step("/a/b_checkpoints/checkpoint-12") == 12
    assert checkpoint_step(None) is None
