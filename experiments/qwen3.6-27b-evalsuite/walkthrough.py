"""Per-token AV walkthrough analysis: at each couplet token, what fraction of
explanations mention (a) the FUTURE rhyme word (the plan) and (b) the visible
topic words? Prints a table per model x tag."""
import json
import re
import sys

PLAN = {"paper": r"\brabbit", "spont": r"\bhouse"}      # future word
VISIBLE = {"paper": r"\bcarrot", "spont": r"\bmouse|\bmice|\bcat\b"}  # in text


def main(d="."):
    meta = json.load(open(f"{d}/po_meta.json"))
    for model in ("mat", "std"):
        av = json.load(open(f"{d}/po_av_{model}.json"))["tags"]
        for tag in ("paper", "spont"):
            pieces = av[tag]["pieces"]
            gens = av[tag]["gens"]
            print(f"\n=== {model} / {tag} "
                  f"(plan={PLAN[tag]!r}, visible={VISIBLE[tag]!r}) ===")
            print(f"{'i':>3} {'token':<12} {'n':>3} {'plan%':>6} {'vis%':>6}")
            for i in range(len(pieces)):
                g = gens[str(i)]
                pl = sum(bool(re.search(PLAN[tag], x, re.I)) for x in g) / len(g)
                vi = sum(bool(re.search(VISIBLE[tag], x, re.I)) for x in g) / len(g)
                print(f"{i:>3} {pieces[i]!r:<12} {len(g):>3} {pl:>6.0%} {vi:>6.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
