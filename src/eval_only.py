"""Evaluate published weights on CPU - no training code, no GPU.

    BERT_MODEL_ID=Arshu-110405/promptguard-bert GUARD_MODEL_ID=Arshu-110405/promptguard-guard \\
        python -m src.eval_only
    python -m src.eval_only --bert-model-id Arshu-110405/promptguard-bert --guard-model-id Arshu-110405/promptguard-guard
    python -m src.eval_only --models guard --quantize-guard     # also dynamic int8 Guard

Downloads the weights from the Hugging Face Hub (HF_TOKEN for private repos; a
local directory works too), evaluates on the official test split built by
src.load_data (built first if missing) and writes the same metrics JSON as
src.evaluate (artifacts/metrics.json "evaluation", artifacts/model_comparison.json).

--quantize-guard adds a row for Guard with dynamic int8 quantization of its
Linear layers (torch.ao.quantization.quantize_dynamic) and reports accuracy,
latency and size before and after, so the cost of quantization is visible.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ensure_data() -> None:
    from src.load_data import META_PATH, SPLIT_PATHS

    if not META_PATH.exists() or not all(p.exists() for p in SPLIT_PATHS.values()):
        print("Official splits not built yet; running src.load_data ...")
        from src import load_data

        load_data.main([])


def _state_dict_mb(model) -> float:
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return round(buffer.tell() / 1e6, 1)


def quantized_guard_rows(test_df, provenance) -> list[dict]:
    """fp32 vs dynamic-int8 Guard on the same rows, plus size."""
    import torch

    from src.evaluate import _evaluate_one
    from src.predict import PromptAnalyzer

    fp32 = PromptAnalyzer(model_type="guard", device="cpu")
    fp32_size = _state_dict_mb(fp32.guard_model)
    fp32_row, _ = _evaluate_one("guard", test_df, "cpu", analyzer=fp32, label="guard")
    int8 = PromptAnalyzer(model_type="guard", device="cpu")
    int8.guard_model = torch.ao.quantization.quantize_dynamic(int8.guard_model, {torch.nn.Linear}, dtype=torch.qint8)
    int8_row, _ = _evaluate_one("guard", test_df, "cpu", analyzer=int8, label="guard_int8")
    int8_size = _state_dict_mb(int8.guard_model)
    fp32_row["model_size_mb"], int8_row["model_size_mb"] = fp32_size, int8_size
    int8_row["quantization"] = {
        "method": "torch.ao.quantization.quantize_dynamic (nn.Linear -> qint8), CPU",
        "binary_accuracy_change": int8_row["binary"]["accuracy"] - fp32_row["binary"]["accuracy"],
        "binary_f1_change": int8_row["binary"]["f1"] - fp32_row["binary"]["f1"],
        "mean_latency_ms_fp32": fp32_row["mean_latency_ms"],
        "mean_latency_ms_int8": int8_row["mean_latency_ms"],
        "size_mb_fp32": fp32_size,
        "size_mb_int8": int8_size,
    }
    return [fp32_row, int8_row]


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=["classical", "bert", "guard"], default=["classical", "bert", "guard"])
    parser.add_argument("--bert-model-id", help="Hub repo id or local path (default: $BERT_MODEL_ID, else models/bert_model).")
    parser.add_argument("--guard-model-id", help="Hub repo id or local path (default: $GUARD_MODEL_ID, else models/guard_model).")
    parser.add_argument("--quantize-guard", action="store_true", help="Also evaluate dynamic int8 Guard and report the cost.")
    args = parser.parse_args(argv)

    if args.bert_model_id:
        os.environ["BERT_MODEL_ID"] = args.bert_model_id
    if args.guard_model_id:
        os.environ["GUARD_MODEL_ID"] = args.guard_model_id
    _ensure_data()

    from src.evaluate import _write_artifacts, evaluate_all, print_comparison
    from src.preprocess import get_splits

    models = [m for m in args.models if not (args.quantize_guard and m == "guard")]
    rows = evaluate_all(models, device="cpu", write_artifacts=False) if models else []
    if args.quantize_guard:
        _, _, test_df, provenance = get_splits()
        try:
            rows += quantized_guard_rows(test_df, provenance)
        except Exception as exc:
            rows.append({"model": "guard", "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print_comparison(rows)
        raise SystemExit("No model could be evaluated; see the skip reasons above.")
    # Same artifacts as src.evaluate. Probabilities are not kept here, so the
    # figures are skipped; the metrics JSON files are identical in shape.
    from src.preprocess import get_splits as _splits

    _, _, test_df, provenance = _splits()
    provenance["evaluated_with"] = "src.eval_only (CPU, downloaded weights)"
    artifacts_dir = ROOT / "artifacts"
    _write_artifacts(rows, {}, test_df, artifacts_dir, artifacts_dir / "figures", provenance)
    print_comparison(rows)
    for r in rows:
        if "quantization" in r:
            q = r["quantization"]
            print(
                f"\nGuard int8 vs fp32: binary accuracy change {q['binary_accuracy_change']:+.4f}, "
                f"F1 change {q['binary_f1_change']:+.4f}, latency {q['mean_latency_ms_fp32']:.1f} -> "
                f"{q['mean_latency_ms_int8']:.1f} ms, size {q['size_mb_fp32']} -> {q['size_mb_int8']} MB"
            )
    print(f"Wrote {artifacts_dir / 'metrics.json'} and {artifacts_dir / 'model_comparison.json'}")
    return rows


if __name__ == "__main__":
    main()
