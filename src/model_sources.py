"""Resolve where the transformer weights (bert, guard) are loaded from.

Priority:
1. ``BERT_MODEL_ID`` / ``GUARD_MODEL_ID`` env var - a Hugging Face Hub repo id
   (private repos need ``HF_TOKEN``) or a local directory.
2. ``models/bert_model`` / ``models/guard_model`` if they hold real weights.

Git LFS pointer stubs in ``models/`` are rejected with a clear message instead
of failing deep inside ``from_pretrained``. This module must stay importable
without torch/transformers so the slim image can report availability.
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# model_type -> (env var, local directory name, training command)
MODEL_SPECS: dict[str, tuple[str, str, str]] = {
    "bert": ("BERT_MODEL_ID", "bert_model", "python -m src.train_bert"),
    "guard": ("GUARD_MODEL_ID", "guard_model", "python -m src.train_guard_model"),
}

_LFS_POINTER_PREFIX = b"version https://git-lfs"
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


class ModelUnavailableError(RuntimeError):
    """A model cannot be loaded. ``code`` is a short, public reason code; the
    message has details (paths, exception text) meant for server logs only."""

    code = "load_failed"


class ModelSourceError(ModelUnavailableError):
    """No usable weights for a transformer model."""

    code = "weights_missing"


class LfsStubError(ModelSourceError):
    """models/<dir> holds Git LFS pointer files instead of weights."""

    code = "lfs_stub"


class RuntimeUnavailableError(ModelUnavailableError):
    """torch/transformers cannot be imported."""

    code = "runtime_unavailable"


class DownloadFailedError(ModelUnavailableError):
    """Weights could not be fetched from the Hugging Face Hub."""

    code = "download_failed"


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def resolve_model_source(model_type: str) -> str:
    """Return a Hub repo id or local path suitable for ``from_pretrained``."""
    env_var, dirname, train_cmd = MODEL_SPECS[model_type]
    override = os.environ.get(env_var, "").strip()
    if override:
        return override

    local_dir = MODELS_DIR / dirname
    hint = f"Set {env_var} to a Hugging Face Hub repo id, or train locally with `{train_cmd}`."
    config = local_dir / "config.json"
    if not config.exists():
        raise ModelSourceError(f"No {model_type} weights: {local_dir} has no config.json. {hint}")
    weights = [local_dir / name for name in _WEIGHT_FILES if (local_dir / name).exists()]
    if _is_lfs_pointer(config) or any(_is_lfs_pointer(w) for w in weights):
        raise LfsStubError(
            f"{local_dir} contains Git LFS pointer stubs, not real weights. "
            f"Run `git lfs pull`, or {hint[0].lower()}{hint[1:]}"
        )
    if not weights:
        raise ModelSourceError(f"No {model_type} weights file in {local_dir}. {hint}")
    return str(local_dir)
