"""Tally poetry-steering results: per tag x arm, rhyme-word distribution,
plan-word rate (original rhyme survives) and target rate (edit succeeded).
Run locally on the pulled po_steer.json."""
import json
import re
import sys
from collections import Counter

CONFIG_TARGETS = {
    "paper": dict(plan=["rabbit", "rabbits", "habit"],
                  target=["mouse", "mice", "house", "cheese"],
                  loose=["mouse", "mice", "house", "cheese", "houses"]),
    "spont": dict(plan=["house", "houses", "mouse"],
                  target=["rabbit", "rabbits", "habit", "carrot"],
                  loose=["rabbit", "rabbits", "habit", "carrot",
                         "habitat", "habitats", "habits"]),
}


def last_words(gens):
    out = []
    for g in gens:
        line = g.split("\n")[0].strip()
        w = re.sub(r"[^a-z]", "", line.split()[-1].lower()) if line.split() else ""
        out.append(w)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "po_steer.json"
    data = json.load(open(path))
    out = {}
    for tag, R in data.items():
        T = CONFIG_TARGETS[tag]
        print(f"\n=== {tag} (n={R['n']}) ===")
        print(f"{'arm':<26} {'plan%':>6} {'target%':>8} {'loose%':>7}  top endings")
        out[tag] = {}
        for name, gens in R["arms"].items():
            ws = last_words(gens)
            n = len(ws)
            plan = sum(w in T["plan"] for w in ws) / n
            targ = sum(w in T["target"] for w in ws) / n
            loose = sum(w in T["loose"] for w in ws) / n
            top = dict(Counter(ws).most_common(4))
            out[tag][name] = dict(plan=round(plan, 3), target=round(targ, 3),
                                  loose=round(loose, 3), top=top)
            print(f"{name:<26} {plan:>6.0%} {targ:>8.0%} {loose:>7.0%}  {top}")
    json.dump(out, open("po_tally.json", "w"), indent=1)
    print("\n[saved] po_tally.json")


if __name__ == "__main__":
    main()
