"""Real-world prompts: jailbreaks and regular prompts from the wild, and real ChatGPT traffic.

Two sources, both pinned and used without retraining:

* in-the-wild (Shen et al., CCS 2024; TrustAIRLab/in-the-wild-jailbreak-prompts, MIT): jailbreak
  prompts and regular prompts that people shared on Reddit, Discord and prompt websites (snapshot
  2023-12-25). All jailbreak prompts (label MALICIOUS) and a fixed-seed sample of the regular prompts
  (label SAFE). Note: these are jailbreaks (the user is the attacker), not indirect injection.
* WildChat (allenai/WildChat-1M, ODC-BY): the first user turn of real ChatGPT conversations from the
  first data shard, English only, a fixed-seed sample. This release contains only conversations that
  the OpenAI Moderation API and Detoxify did not flag as toxic (dataset card). The prompts are UNLABELED
  and may include some jailbreak attempts, so the result is a flag rate on real traffic, not a
  false-positive rate. (flag_rate_non_toxic is kept for releases that do contain toxic rows; here it
  equals flag_rate.)

Texts that also occur in the training data (exact or normalized) are removed first, and duplicates
within a set are dropped. No prompt text is written anywhere: the outputs hold aggregate rates only,
and the probability files hold a text fingerprint, not text. Inference runs on --device (default
cuda if available); no latency is measured here.

Output: artifacts/real_world.json, artifacts/tables/real_world.{md,tex}; probabilities in artifacts/probs/.

    python -m src.real_world_eval --protectai --baselines regex keyword llama_pg2
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import probs_io
from src.metrics import LABELS, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "real_world.json"
TABLES_DIR = ROOT / "artifacts" / "tables"
SEED = 2026
ITW = {"repo": "TrustAIRLab/in-the-wild-jailbreak-prompts", "revision": "a10aab8eff1c73165a442d4464dce192bd28b9c5",
       "jailbreak": "jailbreak_2023_12_25/train-00000-of-00001.parquet",
       "regular": "regular_2023_12_25/train-00000-of-00001.parquet", "license": "MIT"}
WILDCHAT = {"repo": "allenai/WildChat-1M", "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
            "file": "data/train-00000-of-00014.parquet", "license": "ODC-BY"}
REGULAR_SAMPLE = 2000
WILDCHAT_SAMPLE = 2000
MIN_SOURCE_ROWS = 30  # per-source flag rates only for sources with at least this many regular prompts


def _download(repo: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["text"].astype(str).str.strip().str.len() > 0]
    return frame.drop_duplicates("text").reset_index(drop=True)


def load_in_the_wild() -> pd.DataFrame:
    jail = pd.read_parquet(_download(ITW["repo"], ITW["revision"], ITW["jailbreak"]), columns=["prompt", "platform", "source"])
    reg = pd.read_parquet(_download(ITW["repo"], ITW["revision"], ITW["regular"]), columns=["prompt", "platform", "source"])
    jail = _dedupe(jail.rename(columns={"prompt": "text"}).assign(label="MALICIOUS"))
    reg = _dedupe(reg.rename(columns={"prompt": "text"}).assign(label="SAFE"))
    reg = reg[~reg["text"].isin(set(jail["text"]))]  # a prompt listed as both counts as a jailbreak
    reg = reg.sample(n=min(REGULAR_SAMPLE, len(reg)), random_state=SEED)
    return pd.concat([jail, reg], ignore_index=True)


def first_user_turn(conversation) -> tuple[str, bool]:
    for turn in conversation:
        if turn.get("role") == "user":
            return str(turn.get("content") or ""), bool(turn.get("toxic"))
    return "", False


def load_wildchat() -> pd.DataFrame:
    raw = pd.read_parquet(_download(WILDCHAT["repo"], WILDCHAT["revision"], WILDCHAT["file"]),
                          columns=["conversation", "language"])
    raw = raw[raw["language"] == "English"]
    turns = [first_user_turn(list(c)) for c in raw["conversation"]]
    frame = pd.DataFrame({"text": [t for t, _ in turns], "toxic": [x for _, x in turns]})
    frame = _dedupe(frame)
    return frame.sample(n=min(WILDCHAT_SAMPLE, len(frame)), random_state=SEED).reset_index(drop=True).assign(label="UNLABELED")


# --------------------------------------------------------------------------- scoring


def rate(hits: int, n: int) -> dict:
    lo_hi = wilson_interval(hits, n) if n else None
    return {"n": int(n), "rate": hits / n if n else None, "wilson95": list(lo_hi) if lo_hi else None}


def score_in_the_wild(frame: pd.DataFrame, probs: np.ndarray, n_boot: int) -> dict:
    from src import curves

    flagged = np.asarray(probs).argmax(axis=1) != LABELS.index("SAFE")
    attack = (frame["label"] == "MALICIOUS").to_numpy()
    y, score = attack, curves.attack_score(probs)
    out = {
        "jailbreaks_detected": rate(int(flagged[attack].sum()), int(attack.sum())),
        "regular_flagged": rate(int(flagged[~attack].sum()), int((~attack).sum())),
        "accuracy": float((flagged == attack).mean()),
        "auroc": float(curves.point_metrics(y, score)["auroc"]),
        "auroc_ci95": curves.bootstrap(y, score, n_boot)["auroc"],
        "regular_flagged_by_source": {},
    }
    regular = frame[~attack]
    for source, idx in regular.groupby("source").groups.items():
        if len(idx) >= MIN_SOURCE_ROWS:
            out["regular_flagged_by_source"][str(source)] = rate(int(flagged[list(idx)].sum()), len(idx))
    return out


def score_wildchat(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    flagged = np.asarray(probs).argmax(axis=1) != LABELS.index("SAFE")
    non_toxic = ~frame["toxic"].to_numpy()
    return {"flag_rate": rate(int(flagged.sum()), len(flagged)),
            "flag_rate_non_toxic": rate(int(flagged[non_toxic].sum()), int(non_toxic.sum()))}


def table_rows(result: dict) -> list[list[str]]:
    def cell(r: dict | None) -> str:
        if not r or not r.get("n"):
            return "-"
        lo, hi = r["wilson95"]
        return f"{r['rate']:.2f} [{lo:.2f}, {hi:.2f}]"

    rows = []
    for model in result["models"]:
        itw, wc = result["models"][model]["in_the_wild"], result["models"][model]["wildchat"]
        rows.append([model, cell(itw["jailbreaks_detected"]), cell(itw["regular_flagged"]),
                     f"{itw['auroc']:.3f}", cell(wc["flag_rate"]), cell(wc["flag_rate_non_toxic"])])
    return rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protectai", action="store_true")
    parser.add_argument("--baselines", nargs="*", default=[], choices=["regex", "keyword", "llama_pg2"])
    parser.add_argument("--device", default=None, help="torch device for transformer models (default: cuda if available)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    import torch

    from src.external_eval import load_models, remove_training_overlap, training_texts
    from src.paper_results import to_latex, to_markdown
    from src.preprocess import clean_text

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    exact, normalized, provenance = training_texts()
    sets = {}
    for name, loader in (("in_the_wild", load_in_the_wild), ("wildchat", load_wildchat)):
        frame = loader()
        kept, dedup = remove_training_overlap(frame, exact, normalized)
        sets[name] = {"frame": kept, "texts": [clean_text(t) for t in kept["text"]],
                      "info": {"rows_before_overlap_removal": int(len(frame)), **dedup, "rows": int(len(kept)),
                               "label_counts": {str(k): int(v) for k, v in kept["label"].value_counts().items()}}}
        if name == "wildchat":
            sets[name]["info"]["toxic_rows"] = int(kept["toxic"].sum())
        print(f"[{name}] {sets[name]['info']}")

    # One model at a time, freeing GPU memory in between (all of them together do not fit a 4 GB GPU).
    plan = [dict(include_protectai=False, trained_models=(m,)) for m in ("classical", "bert", "guard")]
    if args.protectai:
        plan.append(dict(include_protectai=True, include_trained=False))
    plan += [dict(include_protectai=False, include_trained=False, baseline_names=(b,)) for b in args.baselines]
    skipped: dict = {}
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"in_the_wild": ITW, "wildchat": WILDCHAT},
        "config": {"seed": SEED, "regular_sample": REGULAR_SAMPLE, "wildchat_sample": WILDCHAT_SAMPLE,
                   "wildchat_selection": "first user turn, language == English, deduplicated",
                   "min_source_rows": MIN_SOURCE_ROWS, "device": device, "n_bootstrap": args.bootstrap,
                   "training_dataset": f"{provenance['name']} @ {provenance['revision'][:7]}"},
        "sets": {name: s["info"] for name, s in sets.items()},
        "models_skipped": skipped,
        "models": {},
    }
    for step in plan:
        models, step_skipped = load_models(device=device, **step)
        skipped.update(step_skipped)
        for model, predict in models.items():
            result["models"][model] = {}
            for name, s in sets.items():
                texts = s["texts"]
                probs = np.concatenate([predict(texts[i : i + 8]) for i in range(0, len(texts), 8)])
                probs_io.save_probs(f"real_{name}_{model}", probs, s["frame"]["label"].tolist(), texts)
                result["models"][model][name] = (score_in_the_wild(s["frame"], probs, args.bootstrap)
                                                 if name == "in_the_wild" else score_wildchat(s["frame"], probs))
            print(f"[{model}] {json.dumps({k: v.get('jailbreaks_detected', v.get('flag_rate')) for k, v in result['models'][model].items()})}",
                  flush=True)
        predict = None  # drop the last reference to the model so its GPU memory can be freed
        del models
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    result["models_skipped"] = skipped
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    columns = ["model", "in-the-wild jailbreaks detected", "in-the-wild regular flagged", "AUROC",
               "WildChat flag rate", "WildChat flag rate (non-toxic)"]
    caption = (f"Real-world prompts. In-the-wild (Shen et al.): {result['sets']['in_the_wild']['label_counts']} after "
               f"removing training overlap. WildChat: {result['sets']['wildchat']['rows']} English first user turns "
               "(unlabeled: a flag rate, not an FPR). Rate [Wilson 95% CI].")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "real_world.md").write_text(f"{caption}\n\n" + to_markdown(columns, table_rows(result)), encoding="utf-8")
    (TABLES_DIR / "real_world.tex").write_text(to_latex(columns, table_rows(result), caption, "tab:real-world"), encoding="utf-8")
    print(to_markdown(columns, table_rows(result)))
    return result


if __name__ == "__main__":
    main()
