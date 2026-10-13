"""Upload trained bert/guard weights to a PRIVATE Hugging Face Hub repo.

    python -m src.hub_upload --model bert  --path artifacts/paper_runs/bert/seed_42
    python -m src.hub_upload --model guard --path artifacts/paper_runs/guard/seed_42

Default repos: Arshu-110405/promptguard-bert and Arshu-110405/promptguard-guard.
Needs HF_TOKEN with write access. Before anything is created or uploaded, the
account the token belongs to (huggingface_hub.whoami) must match the repo
namespace; otherwise the upload stops. Repos are private unless --public.

The upload contains the weights, config, tokenizer, metrics.json and a model
card (README.md) built from metrics.json that states the training data and its
licenses. Smoke-test weights and runs that failed the sanity check are refused.
Training scripts use the same path for --push-to-hub.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HF_NAMESPACE = "Arshu-110405"
DEFAULT_REPOS = {"bert": f"{HF_NAMESPACE}/promptguard-bert", "guard": f"{HF_NAMESPACE}/promptguard-guard"}
UPLOAD_PATTERNS = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "metrics.json",
]


class HubUploadError(RuntimeError):
    """A failure with a message meant for the person running the upload."""


def check_namespace(repo_id: str, whoami=None) -> str:
    """Return the token's account name; raise unless it owns *repo_id*'s namespace."""
    if "/" not in repo_id:
        raise HubUploadError(f"Repo id must be <account>/<name>, got {repo_id!r}.")
    if whoami is None:
        from huggingface_hub import whoami
    try:
        account = whoami()["name"]
    except Exception as exc:
        raise HubUploadError(
            f"Could not identify the Hugging Face account ({type(exc).__name__}). "
            "Set HF_TOKEN to a write token for the target account."
        ) from exc
    namespace = repo_id.split("/", 1)[0]
    if namespace != account:
        raise HubUploadError(
            f"HF_TOKEN belongs to {account!r}, but the repo namespace is {namespace!r}. "
            "Stopping: nothing was created or uploaded."
        )
    return account


def _metric(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def model_card(metrics: dict, repo_id: str) -> str:
    """Model card from a run's metrics.json: data, licenses, test results, limits."""
    from src.load_data import LICENSE_INFO, SYNTHETIC_NOTE

    ds = metrics["dataset"]
    hp = metrics["hyperparameters"]
    binary = metrics["test"]["binary"]
    three = metrics["test"]["three_class"]
    declared = ds["license"]["dataset"] if isinstance(ds["license"], dict) else ds["license"]
    sources = "\n".join(f"| {name} | {lic} |" for name, lic in LICENSE_INFO["source_licenses"].items())
    return f"""---
license: other
license_name: see-training-data-licenses
base_model: {metrics['base_model']}
pipeline_tag: text-classification
tags:
- prompt-injection
- promptguard
---

# {repo_id}

PromptGuard `{metrics['model']}` classifier (SAFE / SUSPICIOUS / MALICIOUS), fine-tuned from
`{metrics['base_model']}`. Code: https://github.com/Arshu-1104/AI-Prompt-Injection

## Training data and licenses

- Dataset: `{ds['name']}` @ revision `{ds['revision']}` (config `{ds['train_config']}`), official splits.
- Declared dataset license: {declared}.
- {LICENSE_INFO['caveat']}

| source inside the dataset | license |
|---|---|
{sources}

The weights are derived from this data. Check the source licenses before any commercial use
or redistribution; this repository is private until the maintainer decides on a release.

{SYNTHETIC_NOTE}

## Training

epochs {hp['epochs']}, batch size {hp['batch_size']}, learning rate {hp['learning_rate']},
max length {hp['max_length']} tokens, seed {hp['seed']}, {metrics['precision']} on {metrics['device']};
checkpoint selected on the official validation split.

## Results (official test split)

| view | accuracy | F1 | false-positive rate | ECE | rows |
|---|---|---|---|---|---|
| binary SAFE vs MALICIOUS, original Hub rows | {_metric(binary['accuracy'])} | {_metric(binary['f1'])} | {_metric(binary['false_positive_rate'])} | {_metric(binary['ece'])} | {binary['n_samples']} |
| 3-class (SUSPICIOUS is synthetic) | {_metric(three['accuracy'])} | {_metric(three['f1_weighted'])} (weighted) | - | {_metric(three['ece'])} | {three['n_samples']} |

## Limitations

- In-distribution only. On external prompt-injection test sets (xTRam1, deepset) the model flags a
  large share of benign prompts; it does not generalize beyond its training distribution. See the
  "External evaluation" section of the repository README for the numbers.
- English only; no multilingual support is claimed.
- The SUSPICIOUS class is synthetic and easy to detect; use the binary metrics for comparisons.
"""


def upload_dir(local_dir: Path, repo_id: str, private: bool = True, api=None, whoami=None) -> str:
    """Check the account, create the repo (private by default) and upload *local_dir*."""
    local_dir = Path(local_dir)
    metrics_path = local_dir / "metrics.json"
    if not metrics_path.exists():
        raise HubUploadError(f"{metrics_path} not found; upload only completed training runs.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("smoke_test"):
        raise HubUploadError("Refusing to upload smoke-test weights (--max-samples).")
    if not metrics.get("sanity_check", {}).get("passed", False):
        raise HubUploadError("Refusing to upload a run that failed its sanity check.")

    account = check_namespace(repo_id, whoami=whoami)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    print(f"Account: {account}. Uploading {local_dir} to https://huggingface.co/{repo_id} (private={private}) ...")
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, allow_patterns=UPLOAD_PATTERNS,
                      commit_message="Upload PromptGuard weights")
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(model_card(metrics, repo_id), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id,
                        commit_message="Model card with training-data licenses")
    return account


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(DEFAULT_REPOS), required=True)
    parser.add_argument("--path", required=True, help="Directory with the trained weights and metrics.json.")
    parser.add_argument("--repo", help="Target repo id (default: %s/promptguard-<model>)." % HF_NAMESPACE)
    parser.add_argument("--public", action="store_true", help="Create the repo public (default: private).")
    args = parser.parse_args(argv)
    repo = args.repo or DEFAULT_REPOS[args.model]
    try:
        upload_dir(Path(args.path), repo, private=not args.public)
    except HubUploadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    env_var = {"bert": "BERT_MODEL_ID", "guard": "GUARD_MODEL_ID"}[args.model]
    print(f"Uploaded. Serve it with {env_var}={repo} (and HF_TOKEN while the repo is private).")


if __name__ == "__main__":
    main()
