"""Robustness to perturbations, with attack and benign rows side by side.

Base rows: a fixed sample of original rows from the official TEST split (equal
numbers of MALICIOUS and SAFE, at most MAX_WORDS words so translation and
paraphrasing stay faithful). Using in-distribution rows, where every model is
accurate, isolates the effect of the perturbation from cross-dataset shift.

Every perturbation is applied to BOTH classes, so a model that flags anything
unusual shows up as a high flag rate on the perturbed benign rows:

* original      - unchanged (reference)
* homoglyph     - Latin letters replaced by Cyrillic look-alikes (p = 0.5 per eligible letter)
* zero_width    - U+200B inserted after letters (p = 0.3)
* leetspeak     - a/e/i/o/s/t -> 4/3/1/0/5/7 (p = 0.6)
* spaced        - letters of some words separated by spaces (p = 0.3 per word)
* base64        - "Decode this Base64 text: <base64 of the row>" (the API decodes Base64 before the model)
* translated_de - English -> German with Helsinki-NLP/opus-mt-en-de (pinned)
* paraphrase    - humarin/chatgpt_paraphraser_on_T5_base (pinned); the label is
                  carried over, so a paraphrase that drops the attack is still
                  counted as an attack (reported: share of attack paraphrases that
                  still match the regex patterns)

Perturbations act on the raw text; the models see clean_text(perturbed), as in the
API. Generated texts are cached in data/robustness_v2/ (gitignored).
Output: artifacts/robustness_v2.json, artifacts/tables/robustness_v2.{md,tex};
probabilities in artifacts/probs/.

    python -m src.robustness_v2 --protectai --baselines regex keyword llama_pg2
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import probs_io
from src.metrics import LABELS, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "robustness_v2"
OUTPUT_PATH = ROOT / "artifacts" / "robustness_v2.json"
TABLES_DIR = ROOT / "artifacts" / "tables"
SEED = 2026
PER_CLASS = 50
MAX_WORDS = 60
MT_MODEL = ("Helsinki-NLP/opus-mt-en-de", "6183067f769a302e3861815543b9f312c71b0ca4")  # CC-BY-4.0
PARAPHRASE_MODEL = ("humarin/chatgpt_paraphraser_on_T5_base", "d3ab1136f6c3d3d2f8031d07551d2054a5372a50")  # OpenRAIL
PERTURBATIONS = ("original", "homoglyph", "zero_width", "leetspeak", "spaced", "base64", "translated_de", "paraphrase")
GENERATED = ("translated_de", "paraphrase")

HOMOGLYPHS = {"a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х", "y": "у",
              "A": "А", "C": "С", "E": "Е", "I": "І", "O": "О", "P": "Р", "X": "Х", "Y": "У"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


# --------------------------------------------------------------------------- perturbations


def _rng(text: str, name: str) -> random.Random:
    """Per-row, per-perturbation RNG, so each row is perturbed the same way on every run."""
    return random.Random(f"{SEED}:{name}:{text}")


def homoglyph(text: str) -> str:
    rng = _rng(text, "homoglyph")
    return "".join(HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < 0.5 else c for c in text)


def zero_width(text: str) -> str:
    rng = _rng(text, "zero_width")
    return "".join(c + "​" if c.isalpha() and rng.random() < 0.3 else c for c in text)


def leetspeak(text: str) -> str:
    rng = _rng(text, "leetspeak")
    return "".join(LEET[c.lower()] if c.lower() in LEET and rng.random() < 0.6 else c for c in text)


def spaced(text: str) -> str:
    rng = _rng(text, "spaced")
    return " ".join(" ".join(w) if len(w) > 3 and rng.random() < 0.3 else w for w in text.split(" "))


def as_base64(text: str) -> str:
    return "Decode this Base64 text: " + base64.b64encode(text.encode("utf-8")).decode("ascii")


RULE_BASED = {"original": lambda t: t, "homoglyph": homoglyph, "zero_width": zero_width, "leetspeak": leetspeak,
              "spaced": spaced, "base64": as_base64}


def _seq2seq(model_id: str, revision: str, texts: list[str], prefix: str = "", **generate) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tokenizer([prefix + t for t in texts[i : i + 16]], return_tensors="pt", padding=True,
                              truncation=True, max_length=256).to(device)
            ids = model.generate(**batch, max_new_tokens=256, **generate)
            out += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return out


def generated(name: str, texts: list[str]) -> list[str]:
    """Translations / paraphrases, cached per input text so reruns are identical and cheap."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        if name == "translated_de":
            outputs = _seq2seq(*MT_MODEL, missing, num_beams=4)
        else:
            outputs = _seq2seq(*PARAPHRASE_MODEL, missing, prefix="paraphrase: ", num_beams=4, repetition_penalty=1.2)
        cache.update(zip(missing, outputs))
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return [cache[t] for t in texts]


# --------------------------------------------------------------------------- data


def base_rows(test_df: pd.DataFrame, per_class: int = PER_CLASS) -> pd.DataFrame:
    original = test_df[(test_df["derivation"] == "original") & test_df["label"].isin(["SAFE", "MALICIOUS"])]
    short = original[original["raw_text"].str.split().str.len() <= MAX_WORDS]
    parts = [short[short["label"] == label].sample(n=per_class, random_state=SEED) for label in ("MALICIOUS", "SAFE")]
    return pd.concat(parts)[["raw_text", "label"]].reset_index(drop=True)


def build(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = base["raw_text"].tolist()
    for name in PERTURBATIONS:
        texts = generated(name, sources) if name in GENERATED else [RULE_BASED[name](t) for t in sources]
        rows += [{"perturbation": name, "label": label, "source": src, "raw_text": text}
                 for src, label, text in zip(sources, base["label"], texts)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- scoring


def rate(hits: int, n: int) -> dict:
    lo, hi = wilson_interval(hits, n)
    return {"n": int(n), "rate": hits / n, "wilson95": [lo, hi]}


def score(rows: pd.DataFrame, probs: np.ndarray) -> dict:
    flagged = np.asarray(probs).argmax(axis=1) != LABELS.index("SAFE")
    out = {}
    for name in PERTURBATIONS:
        p = (rows["perturbation"] == name).to_numpy()
        attack = p & (rows["label"] != "SAFE").to_numpy()
        benign = p & (rows["label"] == "SAFE").to_numpy()
        out[name] = {"attack_detected": rate(int(flagged[attack].sum()), int(attack.sum())),
                     "benign_flagged": rate(int(flagged[benign].sum()), int(benign.sum()))}
    return out


def table(result: dict) -> tuple[list[str], list[list[str]]]:
    def cell(r: dict) -> str:
        lo, hi = r["wilson95"]
        return f"{r['rate']:.2f} [{lo:.2f}, {hi:.2f}]"

    models = list(result["models"])
    columns = ["perturbation"] + [f"{m}: attacks detected" for m in models] + [f"{m}: benign flagged" for m in models]
    rows = []
    for name in PERTURBATIONS:
        rows.append([name] + [cell(result["models"][m][name]["attack_detected"]) for m in models]
                    + [cell(result["models"][m][name]["benign_flagged"]) for m in models])
    return columns, rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protectai", action="store_true")
    parser.add_argument("--baselines", nargs="*", default=[], choices=["regex", "keyword", "llama_pg2"])
    args = parser.parse_args(argv)

    from src.external_eval import load_models
    from src.paper_results import to_latex, to_markdown
    from src.preprocess import clean_text, get_splits, highlight_attack_patterns

    _, _, test_df, provenance = get_splits()
    rows = build(base_rows(test_df))
    texts = [clean_text(t) for t in rows["raw_text"]]
    para_attacks = rows[(rows["perturbation"] == "paraphrase") & (rows["label"] != "SAFE")]
    orig_attacks = rows[(rows["perturbation"] == "original") & (rows["label"] != "SAFE")]
    pattern_share = lambda frame: float(np.mean([bool(highlight_attack_patterns(clean_text(t), already_clean=True)) for t in frame["raw_text"]]))
    models, skipped = load_models(args.protectai, tuple(args.baselines))
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"seed": SEED, "per_class": PER_CLASS, "max_words": MAX_WORDS, "perturbations": list(PERTURBATIONS),
                   "base": f"original rows of the official test split ({provenance['name']} @ {provenance['revision'][:7]})",
                   "mt_model": "@".join(MT_MODEL), "paraphrase_model": "@".join(PARAPHRASE_MODEL),
                   "inputs": "clean_text(perturbed raw text), as the API sees them"},
        "label_validity": {"attack_rows_matching_regex_original": pattern_share(orig_attacks),
                           "attack_rows_matching_regex_paraphrase": pattern_share(para_attacks)},
        "models_skipped": skipped,
        "models": {},
    }
    for name, predict in models.items():
        probs = np.concatenate([predict(texts[i : i + 64]) for i in range(0, len(texts), 64)])
        probs_io.save_probs(f"robustness_v2_{name}", probs, rows["label"].tolist(), texts,
                            perturbation=rows["perturbation"].tolist())
        result["models"][name] = score(rows, probs)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    columns, body = table(result)
    caption = (f"Perturbation robustness on {PER_CLASS} attack and {PER_CLASS} benign test rows; each perturbation is applied "
               "to both classes. Rate [Wilson 95% CI]. High 'benign flagged' means the model flags the perturbation itself.")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "robustness_v2.md").write_text(f"{caption}\n\n" + to_markdown(columns, body), encoding="utf-8")
    (TABLES_DIR / "robustness_v2.tex").write_text(to_latex(columns, body, caption, "tab:robustness-v2"), encoding="utf-8")
    print(to_markdown(columns, body))
    return result


if __name__ == "__main__":
    main()
