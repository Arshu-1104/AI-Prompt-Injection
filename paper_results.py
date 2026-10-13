"""Paper-ready comparison of the three PromptGuard models.

Runs the same evaluation as ``src/evaluate.py`` (classical, bert, guard on the
held-out test set, CPU latency) and writes the comparison table as Markdown and
LaTeX to artifacts/paper/.

Usage:
    python paper_results.py
    python paper_results.py --max-samples 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import MODEL_TYPES, comparison_table, evaluate_all, print_comparison

DISPLAY_NAMES = {
    "classical": "TF-IDF + LogReg",
    "bert": "DistilBERT",
    "guard": "ModernBERT-base",
}


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def to_latex(table: pd.DataFrame) -> str:
    def esc(text: str) -> str:
        return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "r" * (len(table.columns) - 1) + "}",
        r"\toprule",
        " & ".join(esc(c) for c in table.columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(_fmt(v)) for v in row) + r" \\" for row in table.itertuples(index=False)]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Held-out test set results; latency is single-prompt CPU inference.}",
        r"\label{tab:models}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the model comparison table for the paper.")
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    rows = evaluate_all(args.models, device=args.device, max_samples=args.max_samples)
    print_comparison(rows)

    table = comparison_table(rows)
    table = table[table["status"] == "ok"].drop(columns=["status"])
    table["n"] = table["n"].astype(int)
    table.insert(1, "architecture", table["model"].map(DISPLAY_NAMES))
    table = table.rename(
        columns={"f1_weighted": "weighted F1", "ece": "ECE", "mean_latency_ms": "CPU latency (ms)"}
    )

    out_dir = Path(__file__).resolve().parent / "artifacts" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_comparison.md").write_text(to_markdown(table), encoding="utf-8")
    (out_dir / "model_comparison.tex").write_text(to_latex(table), encoding="utf-8")
    skipped = [r["model"] for r in rows if r["status"] != "ok"]
    print(f"\nWrote {out_dir / 'model_comparison.md'} and .tex")
    if skipped:
        print(f"WARNING: {skipped} were skipped and are missing from the paper table.")


if __name__ == "__main__":
    main()
