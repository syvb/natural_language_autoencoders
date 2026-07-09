"""v2 dataset normalization.

Three heterogeneous sources -> one internal A/B representation:

  - sycophancy : anthropics/evals sycophancy_on_{nlp_survey,political_typology_quiz}.jsonl
                 native two-choice, answers formatted " (A)"/" (B)", question ends in "Answer:".
  - neuroticism: anthropics/evals persona/neuroticism.jsonl
                 agree/disagree Yes/No statements -> converted to a two-choice (A) Yes / (B) No
                 item, with the Yes/No -> letter assignment randomized per item (seeded) so the
                 letter identity is decorrelated from both the Yes/No surface and the trait label.
  - yellow     : hand-built likes_yellow_caa.json (nrimsky shape, "(A)"/"(B)", no "Answer:" line).

Normalized item == (trait, idx, stem, matching_letter, nonmatching_letter) where `stem` is the
question + Choices block WITHOUT any trailing "Answer:" line. The probe code appends a uniform
"\n\nAnswer: (<letter>" (or a chat-templated equivalent) and reads the activation at the letter.

Matching position is always read per row from answer_matching_behavior -- every source is
position-balanced, so the matching letter is never a fixed A or B.

No experiment code here and nothing v1 is imported.
"""
import json
import os
import random
import re
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# trait -> source files, sample cap (questions), format tag, rng seed
TRAITS = {
    "sycophancy": dict(
        files=[f"{DATA}/sycophancy_on_nlp_survey.jsonl",
               f"{DATA}/sycophancy_on_political_typology_quiz.jsonl"],
        n=2000, fmt="ab", seed=11),
    "neuroticism": dict(
        files=[f"{DATA}/neuroticism.jsonl"],
        n=2000, fmt="yesno", seed=22),
    "yellow": dict(
        files=[f"{HERE}/likes_yellow_1000.json"],
        n=2000, fmt="ab", seed=33),
}

_LETTER = re.compile(r"\(?\s*([AB])\s*\)?")


@dataclass
class ABItem:
    trait: str
    idx: int
    stem: str               # question + Choices block, no trailing "Answer:"
    matching_letter: str    # "A" or "B" -- the behavior-matching choice
    nonmatching_letter: str


def _read(path):
    txt = open(path).read()
    if str(path).endswith(".jsonl"):
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    return json.loads(txt)


def _strip_answer(question):
    """Drop a trailing 'Answer:' line (anthropics sycophancy) so all stems look alike."""
    q = question.rstrip()
    if q.endswith("Answer:"):
        q = q[: -len("Answer:")].rstrip()
    return q


def _ab_item(row):
    """Already-two-choice rows (sycophancy, yellow)."""
    stem = _strip_answer(row["question"])
    matching_letter = _LETTER.search(row["answer_matching_behavior"]).group(1)
    return stem, matching_letter


def _yesno_to_ab(row, rng):
    """persona Yes/No statement -> (A)/(B) item with randomized Yes/No->letter mapping."""
    q = row["question"].rstrip()
    yes_is_a = rng.random() < 0.5
    if yes_is_a:
        block = "Choices:\n(A) Yes\n(B) No"
        yes_letter, no_letter = "A", "B"
    else:
        block = "Choices:\n(A) No\n(B) Yes"
        yes_letter, no_letter = "B", "A"
    stem = f"{q}\n\n{block}"
    amb = row["answer_matching_behavior"].strip()  # "Yes" / "No"
    matching_letter = yes_letter if amb == "Yes" else no_letter
    return stem, matching_letter


def load_trait(name):
    cfg = TRAITS[name]
    rows = []
    for f in cfg["files"]:
        rows += _read(f)
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    rows = rows[: cfg["n"]]
    items = []
    for i, row in enumerate(rows):
        if cfg["fmt"] == "yesno":
            stem, ml = _yesno_to_ab(row, rng)
        else:
            stem, ml = _ab_item(row)
        nl = "B" if ml == "A" else "A"
        items.append(ABItem(name, i, stem, ml, nl))
    return items


def load_all():
    return {name: load_trait(name) for name in TRAITS}


if __name__ == "__main__":
    from collections import Counter
    for name, items in load_all().items():
        bal = Counter(it.matching_letter for it in items)
        print(f"\n=== {name}: {len(items)} questions | matching letter balance {dict(bal)} ===")
        ex = items[0]
        print(f"  matching={ex.matching_letter}  stem:\n    " + ex.stem.replace("\n", "\n    ")[:600])
