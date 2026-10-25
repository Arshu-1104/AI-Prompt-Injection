"""Tables for the end-to-end extension latency run (bench/e2e/e2e_latency.mjs).

Reads artifacts/e2e_latency.json; writes artifacts/tables/e2e_latency.{md,tex}.

    python -m src.e2e_report
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "e2e_latency.json"
TABLES_DIR = ROOT / "artifacts" / "tables"


def rows(data: dict) -> list[list[str]]:
    debounce = data["config"]["debounce_ms"]
    out = []
    for model in data["config"]["models"]:
        m = data["local"].get(model, {})
        if "end_to_end_ms" not in m:
            out.append([f"local, {model}", "-", "-", "-", "-", m.get("skipped", "not run")])
            continue
        e2e, inf = m["end_to_end_ms"], m["server_inference_ms"]
        other = e2e["median"] - debounce - inf["median"]
        out.append([f"local, {model}", f"{e2e['median']:.0f}", f"{e2e['p95']:.0f}", f"{inf['median']:.1f}",
                    f"{other:.0f}", f"{e2e['n']} trials, {m['failed_trials']} failed"])
    hosted = data.get("hosted")
    if hosted:
        w = hosted["warm_end_to_end_ms"]
        out.append(["hosted, classical, warm", f"{w['median']:.0f}", f"{w['p95']:.0f}", "-", "-", f"{w['n']} trials"])
        cold = hosted["cold_ms"]
        out.append(["hosted, classical, cold", f"{cold['ms']:.0f}", "-", "-", "-",
                    f"after {hosted['idle_minutes_before_cold']} min idle" + ("" if cold["ok"] else ", badge timed out")])
    return out


def main() -> None:
    from src.paper_results import to_latex, to_markdown

    data = json.loads(INPUT.read_text(encoding="utf-8"))
    columns = ["setting", "end-to-end median ms", "end-to-end p95 ms", "server inference median ms",
               "other (median - debounce - inference)", "notes"]
    session = data["local"].get("typing_session", {})
    caption = (f"Extension end-to-end latency: last keystroke to final badge, including the {data['config']['debounce_ms']} ms "
               f"debounce; CPU inference on {data['config']['machine']['cpu']}. Scripted typing produced "
               f"{session.get('requests_per_minute', float('nan')):.1f} requests/min against the "
               f"{data['config']['rate_limit_per_minute']}/min per-key limit.")
    body = rows(data)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "e2e_latency.md").write_text(f"{caption}\n\n" + to_markdown(columns, body), encoding="utf-8")
    (TABLES_DIR / "e2e_latency.tex").write_text(to_latex(columns, body, caption, "tab:e2e-latency"), encoding="utf-8")
    print(to_markdown(columns, body))


if __name__ == "__main__":
    main()
