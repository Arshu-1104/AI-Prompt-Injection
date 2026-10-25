"""Regenerate every table, figure and number in the paper with one command.

Steps, in order (each is also runnable on its own):
  results   python -m src.paper_results  (reuses finished seeds; never retrains if weights match)
  external  python -m src.external_eval
  curves    python -m src.curves
  pipeline  python -m src.pipeline_eval
  docs      python -m src.pasted_docs
  robust    python -m src.robustness_v2
  diag      python -m src.diagnosis
  real      python -m src.real_world_eval (in-the-wild jailbreaks + WildChat; GPU if available)
  e2e       node bench/e2e/e2e_latency.mjs, then python -m src.e2e_report  (needs Node + Playwright)
  numbers   python -m src.paper_numbers   -> paper/numbers.tex
  pdf       pdflatex paper/main.tex (twice)  (set PDFLATEX to the pdflatex executable)

Llama Prompt Guard 2 is added to every step when HF_TOKEN is set (the model is gated);
otherwise it is skipped with a message. CPU latency numbers are only comparable when
nothing else heavy runs at the same time, so the steps run one after another.

    python -m src.paper_all                     # everything except e2e and pdf
    python -m src.paper_all --with-e2e --pdf    # everything
    python -m src.paper_all --only curves numbers
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("results", "external", "curves", "pipeline", "docs", "robust", "diag", "real", "e2e", "numbers", "pdf")
# Arguments of the run reported in the paper (see artifacts/paper_results.json "config").
RESULTS_ARGS = ["--guard-seeds", "3", "--guard-batch-size", "8", "--max-length", "384", "--protectai",
                "--reuse-finished-seeds"]


def baselines() -> list[str]:
    names = ["regex", "keyword"]
    if os.environ.get("HF_TOKEN"):
        names.append("llama_pg2")
    else:
        print("HF_TOKEN not set: skipping Llama Prompt Guard 2 (gated model).")
    return names


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    print(f"\n=== {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps (in pipeline order).")
    parser.add_argument("--with-e2e", action="store_true", help="Include the end-to-end extension latency run.")
    parser.add_argument("--pdf", action="store_true", help="Compile paper/main.tex at the end.")
    args = parser.parse_args(argv)

    selected = [s for s in STEPS if (args.only and s in args.only) or (not args.only and s not in ("e2e", "pdf"))]
    if not args.only:
        selected += [s for s, on in (("e2e", args.with_e2e), ("pdf", args.pdf)) if on]
        selected = [s for s in STEPS if s in selected]
    py = [sys.executable, "-X", "utf8", "-m"]
    extra = ["--baselines", *baselines()]
    for step in selected:
        if step == "results":
            run(py + ["src.paper_results", *RESULTS_ARGS, *extra])
        elif step == "external":
            run(py + ["src.external_eval", "--protectai", *extra])
        elif step == "curves":
            run(py + ["src.curves"])
        elif step == "pipeline":
            run(py + ["src.pipeline_eval"])
        elif step == "docs":
            run(py + ["src.pasted_docs", "--protectai", *extra])
        elif step == "robust":
            run(py + ["src.robustness_v2", "--protectai", *extra])
        elif step == "diag":
            run(py + ["src.diagnosis"])
        elif step == "real":
            run(py + ["src.real_world_eval", "--protectai", *extra])
        elif step == "e2e":
            node = shutil.which("node") or "node"
            run([node, "bench/e2e/e2e_latency.mjs"], env={**os.environ, "PROMPTGUARD_PYTHON": sys.executable})
            run(py + ["src.e2e_report"])
        elif step == "numbers":
            run(py + ["src.paper_numbers"])
        elif step == "pdf":
            pdflatex = os.environ.get("PDFLATEX") or shutil.which("pdflatex")
            if not pdflatex:
                raise SystemExit("pdflatex not found: set PDFLATEX to its path.")
            for _ in range(2):
                run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=ROOT / "paper")
    print("\nDone:", ", ".join(selected))


if __name__ == "__main__":
    main()
