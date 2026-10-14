"""Hub upload: account check, refusal rules, private repo and model card. No network."""
from __future__ import annotations

import json

import pytest

from src import hub_upload as hu


def _me(name):
    return lambda: {"name": name}


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs["repo_id"], kwargs))

    def upload_file(self, **kwargs):
        with open(kwargs["path_or_fileobj"], encoding="utf-8") as f:
            self.card = f.read()
        self.calls.append(("upload_file", kwargs["repo_id"], kwargs))


def _run_dir(tmp_path, **overrides):
    metrics = {
        "model": "bert",
        "base_model": "distilbert-base-uncased",
        "smoke_test": False,
        "sanity_check": {"passed": True},
        "dataset": {"name": "neuralchemy/Prompt-injection-dataset", "revision": "7d70432", "train_config": "full",
                    "license": "Apache-2.0"},
        "hyperparameters": {"epochs": 3, "batch_size": 32, "learning_rate": 2e-5, "max_length": 384, "seed": 42},
        "precision": "bf16",
        "device": "GPU",
        "test": {
            "binary": {"accuracy": 0.95, "f1": 0.96, "false_positive_rate": 0.04, "ece": 0.03, "n_samples": 942},
            "three_class": {"accuracy": 0.96, "f1_weighted": 0.96, "ece": 0.03, "n_samples": 1148},
        },
        **overrides,
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path


def test_default_repos_use_the_hf_account():
    assert hu.DEFAULT_REPOS == {"bert": "Arshu-110405/promptguard-bert", "guard": "Arshu-110405/promptguard-guard"}


def test_check_namespace_matches_token_account():
    assert hu.check_namespace("Arshu-110405/promptguard-bert", whoami=_me("Arshu-110405")) == "Arshu-110405"


def test_check_namespace_stops_on_mismatch():
    with pytest.raises(hu.HubUploadError, match="belongs to 'Arshu-1104'"):
        hu.check_namespace("Arshu-110405/promptguard-bert", whoami=_me("Arshu-1104"))


def test_check_namespace_stops_without_token():
    def no_token():
        raise RuntimeError("no token")

    with pytest.raises(hu.HubUploadError, match="Could not identify"):
        hu.check_namespace("Arshu-110405/promptguard-bert", whoami=no_token)


def test_mismatch_creates_and_uploads_nothing(tmp_path):
    api = FakeApi()
    with pytest.raises(hu.HubUploadError):
        hu.upload_dir(_run_dir(tmp_path), "Arshu-110405/promptguard-bert", api=api, whoami=_me("someone-else"))
    assert api.calls == []


@pytest.mark.parametrize("override, message", [({"smoke_test": True}, "smoke"),
                                               ({"sanity_check": {"passed": False}}, "sanity")])
def test_refuses_smoke_and_failed_runs(tmp_path, override, message):
    api = FakeApi()
    with pytest.raises(hu.HubUploadError, match=message):
        hu.upload_dir(_run_dir(tmp_path, **override), "Arshu-110405/promptguard-bert", api=api,
                      whoami=_me("Arshu-110405"))
    assert api.calls == []


def test_upload_is_private_and_card_states_licenses(tmp_path):
    api = FakeApi()
    hu.upload_dir(_run_dir(tmp_path), "Arshu-110405/promptguard-bert", api=api, whoami=_me("Arshu-110405"))
    create = api.calls[0]
    assert create[0] == "create_repo" and create[2]["private"] is True
    assert [c[0] for c in api.calls] == ["create_repo", "upload_folder", "upload_file"]
    assert "training_args.bin" not in api.calls[1][2]["allow_patterns"]
    card = api.card
    assert "neuralchemy/Prompt-injection-dataset" in card and "7d70432" in card
    assert "CC BY 4.0" in card and "Research" in card  # source licenses listed
    assert "0.9500" in card  # numbers come from metrics.json
    assert "does not generalize" in card
