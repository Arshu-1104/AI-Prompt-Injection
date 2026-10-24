"""Why do the models fail out of distribution? Sentence form by label and by outcome.

Hypothesis (README): the training data's attacks are mostly imperative and its
benign rows mostly questions, so the models learn "imperative = attack", and
benign imperative task prompts in other datasets become false positives.

This measures, with a transparent heuristic (lists below):
* imperative - some clause (split at punctuation and line breaks) starts, after an
               optional "please", with a base-form verb in IMPERATIVE_VERBS
* question   - the text ends with "?" or starts with a word in QUESTION_WORDS
the share of each form
1. per label in the training split (original rows), and
2. among each external set's true-SAFE rows, split by the classical model's outcome
   at its fixed operating point (false positive vs true negative),
with Wilson 95% intervals. Reads the probability files of src.external_eval.
Output: artifacts/diagnosis.json, artifacts/tables/diagnosis.{md,tex}.

    python -m src.diagnosis
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import probs_io
from src.metrics import LABELS, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "diagnosis.json"
TABLES_DIR = ROOT / "artifacts" / "tables"
MODEL = "classical"

IMPERATIVE_VERBS = frozenset("""
act add analyze answer apply arrange ask assume become break build calculate change check choose classify combine
compare compose convert copy correct count create decide decode define delete describe design determine develop
disregard do draft draw edit enter estimate evaluate explain extract fill find fix follow forget format generate
give go help identify ignore imagine implement improve include insert interpret keep label let list locate make
match name note offer organize output paraphrase pick play predict pretend print produce provide put rank read
recommend reformat remove rephrase replace reply repeat respond restate return reveal review rewrite rate run say
select send share show simplify solve sort speak spell start state stop suggest summarise summarize take talk tell
think transform translate treat turn use verify write
""".split())
QUESTION_WORDS = frozenset("what how why who whom whose when where which can could is are am was were do does did "
                           "should would will shall may might have has".split())


def first_word(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    if words and words[0] == "please":
        words = words[1:]
    return words[0] if words else ""


def sentence_form(text: str) -> str:
    """'question', 'imperative' or 'other' (question wins when both apply).

    Imperative: some clause (split at . , ; : ! ? and line breaks) starts with a listed
    verb, which also catches task prompts such as "Given two words, output ...".
    """
    stripped = str(text).strip()
    if stripped.endswith("?") or first_word(stripped) in QUESTION_WORDS:
        return "question"
    if any(first_word(clause) in IMPERATIVE_VERBS for clause in re.split(r"[.,;:!?\n]", stripped)):
        return "imperative"
    return "other"


def shares(texts) -> dict:
    forms = [sentence_form(t) for t in texts]
    n = len(forms)
    out = {"n": n}
    for form in ("imperative", "question", "other"):
        k = sum(f == form for f in forms)
        out[form] = {"share": k / n if n else None, "wilson95": list(wilson_interval(k, n)) if n else None}
    return out


def main() -> dict:
    from src.external_eval import DATASETS, load_external, remove_training_overlap, training_texts
    from src.paper_results import to_latex, to_markdown
    from src.preprocess import clean_text, get_splits

    train, _, _, provenance = get_splits()
    original = train[train["derivation"] == "original"]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "heuristic": {"imperative": "a clause (split at . , ; : ! ? and newlines) starts with a word in IMPERATIVE_VERBS "
                                    "(after optional 'please')",
                      "question": "ends with '?' or first word in QUESTION_WORDS", "model": MODEL,
                      "imperative_verbs": sorted(IMPERATIVE_VERBS), "question_words": sorted(QUESTION_WORDS)},
        "training": {label: shares(original.loc[original["label"] == label, "raw_text"]) for label in ("SAFE", "MALICIOUS")},
        "training_dataset": f"{provenance['name']} @ {provenance['revision'][:7]} (train split, original rows)",
        "external_true_safe": {},
    }
    exact, normalized, _ = training_texts()
    for ds in DATASETS:
        path = probs_io.PROBS_DIR / f"external_{probs_io.slug(ds.name)}_{MODEL}.npz"
        if not path.exists():
            continue
        try:
            kept, _ = remove_training_overlap(load_external(ds), exact, normalized)
        except Exception as exc:
            print(f"{ds.name} skipped: {type(exc).__name__}: {exc}")
            continue
        raw = kept["text"].astype(str).tolist()
        probs = probs_io.load_probs(path, [clean_text(t) for t in raw])["probs"]
        safe = np.asarray(kept["label"] == "SAFE")
        flagged = probs.argmax(axis=1) != LABELS.index("SAFE")
        result["external_true_safe"][ds.name] = {
            "false_positives": shares([t for t, s, f in zip(raw, safe, flagged) if s and f]),
            "true_negatives": shares([t for t, s, f in zip(raw, safe, flagged) if s and not f]),
        }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def cell(entry: dict, form: str) -> str:
        e = entry[form]
        return "-" if e["share"] is None else f"{e['share']:.2f} [{e['wilson95'][0]:.2f}, {e['wilson95'][1]:.2f}]"

    rows = [[f"training, {label}", str(s["n"]), cell(s, "imperative"), cell(s, "question")]
            for label, s in result["training"].items()]
    for name, d in result["external_true_safe"].items():
        for outcome in ("false_positives", "true_negatives"):
            s = d[outcome]
            rows.append([f"{name}, benign {outcome.replace('_', ' ')}", str(s["n"]), cell(s, "imperative"), cell(s, "question")])
    columns = ["rows", "n", "imperative share", "question share"]
    caption = (f"Sentence form by label (training) and by the {MODEL} model's outcome on benign external rows. "
               "Heuristic and word lists in src/diagnosis.py. Share [Wilson 95% CI].")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "diagnosis.md").write_text(f"{caption}\n\n" + to_markdown(columns, rows), encoding="utf-8")
    (TABLES_DIR / "diagnosis.tex").write_text(to_latex(columns, rows, caption, "tab:diagnosis"), encoding="utf-8")
    print(to_markdown(columns, rows))
    return result


if __name__ == "__main__":
    main()
