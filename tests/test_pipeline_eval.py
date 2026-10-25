from __future__ import annotations

import numpy as np

from src import pipeline_eval as pe

# classical rows: confident SAFE, unsure SUSPICIOUS, confident SUSPICIOUS, unsure MALICIOUS
CLASSICAL = np.array([[0.90, 0.05, 0.05], [0.30, 0.50, 0.20], [0.10, 0.80, 0.10], [0.35, 0.20, 0.45]])
GUARD = np.array([[0.10, 0.10, 0.80], [0.90, 0.05, 0.05], [0.90, 0.05, 0.05], [0.90, 0.05, 0.05]])


def test_deployed_rule_only_escalates_unsure_suspicious():
    assert pe.escalate_mask(CLASSICAL, "suspicious_only", 0.65).tolist() == [False, True, False, False]
    assert pe.escalate_mask(CLASSICAL, "any_label", 0.65).tolist() == [False, True, False, True]
    assert not pe.escalate_mask(CLASSICAL, "suspicious_only", 0.35).any()


def test_pipeline_takes_guard_label_only_where_escalated():
    mask = pe.escalate_mask(CLASSICAL, "suspicious_only", 0.65)
    assert pe.pipeline_labels(CLASSICAL, GUARD, mask).tolist() == [0, 0, 1, 2]


def test_rule_floor_raises_only_safe():
    labels = np.array([0, 1, 2, 0])
    flagged = np.array([True, True, True, False])
    assert pe.apply_rule_floor(labels, flagged).tolist() == [1, 1, 2, 0]


def test_binary_metrics_counts_suspicious_as_attack():
    m = pe.binary_metrics(["SAFE", "SUSPICIOUS", "MALICIOUS", "SAFE"], np.array([0, 1, 0, 2]))
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["false_positive_rate"] == 0.5


def test_threshold_choice_prefers_accuracy_then_fewer_escalations():
    sweep = {"suspicious_only": {
        "0.5": {"accuracy": 0.90, "escalated_share": 0.00},
        "0.6": {"accuracy": 0.92, "escalated_share": 0.10},
        "0.7": {"accuracy": 0.92, "escalated_share": 0.30},
    }}
    assert pe.choose_threshold(sweep, "suspicious_only") == 0.6


def test_pipeline_latency_adds_guard_time_only_for_escalated_texts():
    lat = pe.pipeline_latency(np.array([1.0, 1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0, 10.0]), CLASSICAL)
    assert lat["classical_only_mean_ms"] == 1.0
    assert lat["suspicious_only"]["0.65"] == 1.0 + 10.0 * 1 / 4
    assert lat["any_label"]["0.65"] == 1.0 + 10.0 * 2 / 4
