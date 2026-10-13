"""Shared fine-tuning loop for the DistilBERT (bert) and ModernBERT (guard) models.

Both ``src/train_bert.py`` and ``src/train_guard_model.py`` are thin wrappers
around :func:`train_model`. Run either with ``--help`` for the CLI flags.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.metrics import LABELS, evaluate_split
from src.preprocess import clean_text, get_splits
from src.sanity_prompts import TEST_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class TrainingError(RuntimeError):
    """A failure with a message meant for the person running the script."""


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "bert" | "guard" - the API model id
    base_model: str  # Hugging Face Hub id of the pretrained encoder
    output_dirname: str  # models/<output_dirname>/
    metadata: dict = field(default_factory=dict)


def build_arg_parser(spec: ModelSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Fine-tune {spec.base_model} as the '{spec.name}' model.")
    parser.add_argument("--epochs", type=float, default=3, help="Training epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-device train/eval batch size (default: 32 with a GPU, 16 on CPU).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for init, shuffling and sampling (default: 42).")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Peak learning rate (default: 5e-5).")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max tokens per prompt (default: 128). The share of truncated rows is reported.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Smoke test: cap the train and test sets at N rows each. Output goes to "
            f"artifacts/smoke/{spec.output_dirname}/ so the real weights in models/ are never overwritten, "
            "and a failed sanity check only warns."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Where to save the model (default: models/{spec.output_dirname}).")
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help="After a successful run, upload model, tokenizer, metrics.json and a model card to this Hugging Face Hub "
             "repo, e.g. Arshu-110405/promptguard-bert (needs HF_TOKEN with write access; the token's account must own "
             "the repo namespace, checked before training).",
    )
    parser.add_argument("--hub-public", action="store_true", help="Create the Hub repo as public (default: private).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Where epoch checkpoints go (default: <output-dir>_checkpoints). On Colab/Kaggle point "
                             "this at persistent storage such as Google Drive so --resume works after a disconnect.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --checkpoint-dir.")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep checkpoints after a successful run.")
    parser.add_argument("--early-stopping-patience", type=int, default=1,
                        help="Stop after this many epochs without a better validation F1 (default: 1).")
    return parser


def run_cli(main: Callable[[], None]) -> None:
    """Run a training ``main`` and turn TrainingError into a clean non-zero exit."""
    try:
        main()
    except TrainingError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------- errors


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_network_error(exc: BaseException) -> bool:
    network_types: list[type] = [ConnectionError, TimeoutError]
    try:
        import requests

        network_types.append(requests.exceptions.ConnectionError)
        network_types.append(requests.exceptions.Timeout)
    except Exception:
        pass
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

        network_types += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:
        pass
    markers = ("couldn't connect", "connection error", "max retries exceeded", "name resolution", "offline mode")
    for item in _exception_chain(exc):
        if isinstance(item, tuple(network_types)):
            return True
        if any(marker in str(item).lower() for marker in markers):
            return True
    return False


def _is_oom(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        if isinstance(item, MemoryError) or type(item).__name__ == "OutOfMemoryError":
            return True
        message = str(item).lower()
        if "out of memory" in message or "can't allocate memory" in message or "not enough memory" in message:
            return True
    return False


def _oom_message(spec: ModelSpec, device: str, args: argparse.Namespace) -> str:
    return (
        f"Out of memory while training '{spec.name}' ({spec.base_model}) on {device} with "
        f"--batch-size {args.batch_size} and --max-length {args.max_length}. Retry with "
        f"--batch-size {max(args.batch_size // 2, 1)} and/or --max-length 128, or use a machine with more memory."
    )


# --------------------------------------------------------------------------- data


def _import_runtime():
    try:
        import torch
        import transformers
    except Exception as exc:  # ImportError, or OSError when a DLL is blocked
        raise TrainingError(
            f"PyTorch/transformers could not be imported ({type(exc).__name__}: {exc}). "
            "Install them with `pip install torch --index-url https://download.pytorch.org/whl/cpu` "
            "and `pip install -r requirements-heavy.txt`."
        ) from exc
    return torch, transformers


def _cap_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample at most *n* rows, keeping every label that is present."""
    if len(df) <= n:
        return df
    parts = []
    for _, group in df.groupby("label", sort=False):
        k = max(1, round(n * len(group) / len(df)))
        parts.append(group.sample(n=min(k, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _load_splits(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Official train / validation / test splits; --seed never changes them."""
    try:
        train_df, val_df, test_df, provenance = get_splits()
    except (FileNotFoundError, ValueError) as exc:
        raise TrainingError(str(exc)) from exc
    for df in (train_df, val_df, test_df):
        unknown = set(df["label"]) - set(LABELS)
        if unknown:
            raise TrainingError(f"Unknown labels in dataset: {sorted(unknown)}")
    if args.max_samples:
        train_df = _cap_rows(train_df, args.max_samples, args.seed)
        val_df = _cap_rows(val_df, args.max_samples, args.seed)
        test_df = _cap_rows(test_df, args.max_samples, args.seed)
        provenance["smoke_cap"] = {
            "max_samples": args.max_samples,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        }
    if test_df.empty or val_df.empty:
        raise TrainingError("Validation or test split is empty; cannot evaluate.")
    return train_df, val_df, test_df, provenance


def _encode(tokenizer, df: pd.DataFrame, max_length: int) -> list[dict]:
    encodings = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
    labels = df["label"].map(LABEL_TO_ID).tolist()
    keys = list(encodings.keys())
    return [{**{k: encodings[k][i] for k in keys}, "labels": labels[i]} for i in range(len(labels))]


def _load_base_model(spec: ModelSpec, transformers):
    try:
        # Config first: it is where an unsupported architecture (old transformers) surfaces.
        config = transformers.AutoConfig.from_pretrained(
            spec.base_model, num_labels=len(LABELS), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(spec.base_model)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(spec.base_model, config=config)
    except Exception as exc:
        if "does not recognize this architecture" in str(exc) or isinstance(exc, KeyError):
            raise TrainingError(
                f"transformers {transformers.__version__} cannot load {spec.base_model}. "
                "ModernBERT needs transformers>=4.48 (see requirements-heavy.txt)."
            ) from exc
        if _is_network_error(exc):
            raise TrainingError(
                f"Could not download {spec.base_model} from the Hugging Face Hub: no internet connection "
                "or the Hub is unreachable. Connect and retry, or pre-download it on a connected machine with "
                f"`huggingface-cli download {spec.base_model}` and run with HF_HUB_OFFLINE=1."
            ) from exc
        if _is_oom(exc):
            raise TrainingError(f"Out of memory while loading {spec.base_model}.") from exc
        raise
    return tokenizer, model


# --------------------------------------------------------------------------- inference helpers


def _predict_probs(torch, model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float()
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, len(LABELS)))


def run_sanity_check(torch, model, tokenizer, max_length: int) -> dict:
    """Predict the hand-picked tier prompts; flag a model that collapsed to one class."""
    rows = [(category, prompt) for category, prompts in TEST_PROMPTS.items() for prompt in prompts]
    texts = [clean_text(prompt) for _, prompt in rows]  # same normalization as PromptAnalyzer.predict
    probs = _predict_probs(torch, model, tokenizer, texts, max_length, batch_size=32)
    predictions = []
    print(f"\n--- Sanity check on {len(rows)} prompts from src/sanity_prompts.py ---")
    for (category, prompt), row in zip(rows, probs):
        label = LABELS[int(row.argmax())]
        predictions.append({"category": category, "prompt": prompt, "label": label, "confidence": float(row.max())})
        print(f"  {label:<10} {row.max():.2f}  [{category}] {prompt[:60]}")
    counts = Counter(p["label"] for p in predictions)
    benign = [p for p in predictions if p["category"].startswith("benign")]
    attacks = [p for p in predictions if not p["category"].startswith("benign")]
    result = {
        "n_prompts": len(predictions),
        "prediction_counts": dict(counts),
        "benign_marked_safe": sum(p["label"] == "SAFE" for p in benign) / max(len(benign), 1),
        "attacks_flagged": sum(p["label"] != "SAFE" for p in attacks) / max(len(attacks), 1),
        "passed": len(counts) > 1,
        "predictions": predictions,
    }
    print(
        f"  counts={dict(counts)}  benign->SAFE={result['benign_marked_safe']:.0%}  "
        f"attacks flagged={result['attacks_flagged']:.0%}"
    )
    return result


TRUNCATION_WARN_PCT = 2.0
NAN_CHECK_STEPS = 50


def token_length_report(tokenizer, texts: list[str], max_length: int) -> dict:
    """Token-length distribution of the training texts and the share that --max-length truncates."""
    lengths = np.array([len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]])
    report = {
        "max_length": max_length,
        **{f"p{q}": int(np.percentile(lengths, q)) for q in (50, 90, 95, 99)},
        "max": int(lengths.max()),
        "truncated_pct": float((lengths > max_length).mean() * 100),
    }
    print(
        f"token lengths: p50={report['p50']} p90={report['p90']} p95={report['p95']} p99={report['p99']} "
        f"max={report['max']}; {report['truncated_pct']:.2f}% of rows exceed --max-length {max_length}"
    )
    if report["truncated_pct"] > TRUNCATION_WARN_PCT:
        print(f"WARNING: more than {TRUNCATION_WARN_PCT}% of training rows are truncated; consider a larger --max-length.")
    return report


def _callbacks(transformers):
    """NaN-loss guard and per-epoch wall-clock timer."""

    class NaNGuard(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and state.global_step <= NAN_CHECK_STEPS and not math.isfinite(float(loss)):
                raise TrainingError(
                    f"Loss is {loss} at step {state.global_step} (within the first {NAN_CHECK_STEPS} steps). "
                    "Try a lower --learning-rate or fp32 (unset CUDA mixed precision)."
                )

    class EpochTimer(transformers.TrainerCallback):
        def __init__(self):
            self.epoch_seconds: list[float] = []
            self._start = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self._start = time.monotonic()

        def on_epoch_end(self, args, state, control, **kwargs):
            if self._start is not None:
                self.epoch_seconds.append(round(time.monotonic() - self._start, 1))

    return NaNGuard(), EpochTimer()


# --------------------------------------------------------------------------- training


def _now_iso() -> str:
    """Local wall-clock time, ISO 8601 with the UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def checkpoint_step(name: str | None) -> int | None:
    """'checkpoint-2113' (or a path ending in it) -> 2113."""
    if not name:
        return None
    return int(Path(name).name.rsplit("-", 1)[-1])


def training_provenance(
    *,
    started_at: str,
    finished_at: str,
    global_step: int,
    steps_per_epoch: int,
    best_model_checkpoint: str | None,
    log_history: list[dict],
    resumed_from_step: int | None,
) -> dict:
    """Run facts recorded when training happens (the checkpoint folder is deleted later).

    best_epoch is the epoch of the validation evaluation at the best checkpoint's
    step; None when no checkpoint was selected (smoke runs).
    """
    best_name = Path(best_model_checkpoint).name if best_model_checkpoint else None
    best_step = checkpoint_step(best_name)
    best_epoch = None
    if best_step is not None:
        evals = [e for e in log_history if "eval_f1" in e and e.get("step") == best_step]
        epoch = evals[-1]["epoch"] if evals else best_step / steps_per_epoch
        best_epoch = int(round(float(epoch)))
    return {
        "training_started_at": started_at,
        "training_finished_at": finished_at,
        "total_optimizer_steps": int(global_step),
        "steps_per_epoch": int(steps_per_epoch),
        "best_epoch": best_epoch,
        "best_checkpoint_name": best_name,
        "resumed": resumed_from_step is not None,
        "resumed_from_step": resumed_from_step,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _check_hub_account(repo_id: str) -> None:
    """Before training: the HF_TOKEN account must own the target repo's namespace."""
    from src.hub_upload import HubUploadError, check_namespace

    try:
        account = check_namespace(repo_id)
    except HubUploadError as exc:
        raise TrainingError(f"--push-to-hub {repo_id}: {exc}") from exc
    print(f"--push-to-hub: HF_TOKEN account {account!r} owns {repo_id}")


def _push_to_hub(spec: ModelSpec, args: argparse.Namespace, output_dir: Path) -> None:
    from src.hub_upload import HubUploadError, upload_dir

    private = not args.hub_public
    try:
        # Weights, tokenizer, metrics.json and a model card with the data licenses.
        upload_dir(output_dir, args.push_to_hub, private=private)
    except HubUploadError as exc:
        raise TrainingError(f"Upload to {args.push_to_hub} stopped: {exc} The model is still saved locally.") from exc
    except Exception as exc:
        if _is_network_error(exc):
            raise TrainingError(f"Upload to {args.push_to_hub} failed: Hugging Face Hub unreachable.") from exc
        raise TrainingError(
            f"Upload to {args.push_to_hub} failed ({type(exc).__name__}: {exc}). "
            "Check that HF_TOKEN is set to a token with write access. The model is still saved locally."
        ) from exc
    env_var = {"bert": "BERT_MODEL_ID", "guard": "GUARD_MODEL_ID"}[spec.name]
    print(f"Uploaded. Serve it with {env_var}={args.push_to_hub}")


def train_model(spec: ModelSpec, args: argparse.Namespace, shared_outputs: bool = True) -> dict:
    """Fine-tune, evaluate, sanity-check and save. With shared_outputs=False
    (multi-seed runs) metrics are written only into the output directory, never
    to artifacts/<name>_metrics.json or artifacts/metrics.json."""
    smoke = bool(args.max_samples)
    if smoke and args.push_to_hub:
        raise TrainingError("--push-to-hub cannot be combined with --max-samples; smoke-test weights are not fit to serve.")
    if args.push_to_hub:
        _check_hub_account(args.push_to_hub)  # fail before spending time on training

    torch, transformers = _import_runtime()
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    if smoke:
        artifacts_dir = ROOT / "artifacts" / "smoke"
        output_dir = args.output_dir or artifacts_dir / spec.output_dirname
    else:
        artifacts_dir = ROOT / "artifacts"
        output_dir = args.output_dir or ROOT / "models" / spec.output_dirname
    output_dir = Path(output_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir.parent / f"{output_dir.name}_checkpoints"

    transformers.set_seed(args.seed)
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU"
    if args.batch_size is None:
        args.batch_size = 32 if cuda else 16
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16
    precision = "bf16" if bf16 else "fp16" if fp16 else "fp32"
    train_df, val_df, test_df, provenance = _load_splits(args)
    print(
        f"[{spec.name}] base={spec.base_model} device={device} torch={torch.__version__} "
        f"transformers={transformers.__version__}\n"
        f"[{spec.name}] train={len(train_df)} validation={len(val_df)} test={len(test_df)} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.learning_rate} seed={args.seed} precision={precision} max_length={args.max_length}"
        + (f"  SMOKE TEST (--max-samples {args.max_samples})" if smoke else "")
    )
    print(f"[{spec.name}] train label counts: {train_df['label'].value_counts().to_dict()}")

    tokenizer, model = _load_base_model(spec, transformers)
    lengths = token_length_report(tokenizer, train_df["text"].tolist(), args.max_length)
    train_items = _encode(tokenizer, train_df, args.max_length)
    # Checkpoint selection uses the official validation split; test is only for the final report.
    eval_items = _encode(tokenizer, val_df, args.max_length)

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        from sklearn.metrics import accuracy_score, f1_score

        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="epoch",
        # Smoke runs skip checkpointing: ModernBERT checkpoints are ~1.8 GB each with optimizer state.
        save_strategy="no" if smoke else "epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_steps=10,
        logging_first_step=True,
        group_by_length=True,  # with dynamic padding: batches of similar length, less padding
        bf16=bf16,
        fp16=fp16,
        report_to=[],
    )
    nan_guard, epoch_timer = _callbacks(transformers)
    callbacks = [nan_guard, epoch_timer]
    if not smoke:
        callbacks.append(transformers.EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_items,
        eval_dataset=eval_items,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    resumed_from_step = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint

        # The checkpoint trainer.train(resume_from_checkpoint=True) picks up.
        last = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.is_dir() else None
        resumed_from_step = checkpoint_step(last)
    started_at = _now_iso()
    started = time.monotonic()
    try:
        try:
            trainer.train(resume_from_checkpoint=True if args.resume else None)
        except ValueError as exc:
            if args.resume and "checkpoint" in str(exc).lower():
                raise TrainingError(f"--resume: no checkpoint found in {checkpoint_dir} ({exc}).") from exc
            raise
        finished_at = _now_iso()
        run_facts = training_provenance(
            started_at=started_at,
            finished_at=finished_at,
            global_step=trainer.state.global_step,
            steps_per_epoch=len(trainer.get_train_dataloader()) // training_args.gradient_accumulation_steps,
            best_model_checkpoint=trainer.state.best_model_checkpoint,
            log_history=trainer.state.log_history,
            resumed_from_step=resumed_from_step,
        )
        model = trainer.model
        probs = _predict_probs(torch, model, tokenizer, test_df["text"].tolist(), args.max_length, args.batch_size)
    except Exception as exc:
        if _is_oom(exc):
            raise TrainingError(_oom_message(spec, device, args)) from exc
        raise
    train_seconds = time.monotonic() - started

    test_metrics = evaluate_split(test_df, probs)
    metrics = test_metrics["three_class"]
    sanity = run_sanity_check(torch, model, tokenizer, args.max_length)

    report = {
        "test": test_metrics,
        "model": spec.name,
        "base_model": spec.base_model,
        **spec.metadata,
        "smoke_test": smoke,
        "dataset": provenance,
        "train_samples": len(train_df),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "device": device,
        "precision": precision,
        "token_lengths": lengths,
        "train_seconds": round(train_seconds, 1),
        "timing": {"epoch_seconds": epoch_timer.epoch_seconds, "total_seconds": round(train_seconds, 1)},
        "early_stopping": None if smoke else {
            "patience": args.early_stopping_patience,
            "epochs_completed": float(trainer.state.epoch or 0),
            "best_validation_f1": trainer.state.best_metric,
        },
        **run_facts,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "sanity_check": sanity,
    }
    print(
        f"\n[{spec.name}] accuracy={metrics['accuracy']:.4f} f1_weighted={metrics['f1_weighted']:.4f} "
        f"ece={metrics['ece']:.4f} per_class_f1={ {k: round(v, 3) for k, v in metrics['per_class_f1'].items()} }\n"
        f"[{spec.name}] binary (original Hub rows): accuracy={test_metrics['binary']['accuracy']:.4f} "
        f"f1={test_metrics['binary']['f1']:.4f} fpr={test_metrics['binary']['false_positive_rate']}"
    )

    artifact_metrics_path = (
        artifacts_dir / f"{spec.output_dirname}_metrics.json" if shared_outputs else output_dir / "metrics.json"
    )
    if not sanity["passed"]:
        only = next(iter(sanity["prediction_counts"]))
        message = (
            f"SANITY CHECK FAILED: '{spec.name}' predicted {only} for all {sanity['n_prompts']} sanity prompts. "
            "The model has collapsed to a single class."
        )
        _write_json(artifact_metrics_path, report)
        if not smoke:
            raise TrainingError(f"{message} Weights were NOT saved to {output_dir}. Metrics: {artifact_metrics_path}")
        print(f"\n{'!' * 78}\nWARNING: {message}\nExpected for a tiny smoke run; a full run would stop here.\n{'!' * 78}")

    # Inference reads this so it truncates exactly like training did.
    trainer.model.config.promptguard_max_length = args.max_length
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _write_json(output_dir / "metrics.json", report)
    _write_json(artifact_metrics_path, report)
    if shared_outputs and not smoke:
        combined_path = artifacts_dir / "metrics.json"
        combined = json.loads(combined_path.read_text(encoding="utf-8")) if combined_path.exists() else {}
        combined[spec.output_dirname] = {k: v for k, v in report.items() if k != "sanity_check"}
        _write_json(combined_path, combined)
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(f"[{spec.name}] saved model + tokenizer + metrics.json to {output_dir}")
    print(f"[{spec.name}] metrics: {artifact_metrics_path}")

    if args.push_to_hub:
        _push_to_hub(spec, args, output_dir)
    return report
