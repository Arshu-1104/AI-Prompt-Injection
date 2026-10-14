import numpy as np
import pandas as pd

from src import external_diagnostics as xd


def test_language_heuristic():
    assert xd.language_of("What is the capital of France and how do you get there?") == "en"
    assert xd.language_of("Warum hat Russland die Ukraine überfallen und was ist das Ziel?") == "de"
    assert xd.language_of("E-scooter Germany") == "unknown"


def test_diagnose_reports_inverted_accuracy_and_errors():
    frame = pd.DataFrame({
        "text": ["benign one", "attack one", "benign two", "attack two"],
        "raw_label": [0, 1, 0, 1],
        "label": ["SAFE", "MALICIOUS", "SAFE", "MALICIOUS"],
    })
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1]])
    out = xd.diagnose(frame, probs)
    assert out["metrics"]["accuracy"] == 0.5
    assert out["inverted_label_accuracy"] == 0.5
    assert [fp["text"] for fp in out["top_false_positives"]] == ["benign two"]
    assert [fn["text"] for fn in out["top_false_negatives"]] == ["attack two"]
