"""Why does the matryoshka reveal traits at smaller word budgets? Mechanical
decomposition using the cached reveal judgments (sa_revealcurve_raw.json)
plus the raw explanations — no API calls.

A  reveal onset: among explanations that reveal at ANY budget, the smallest
   word budget at which they reveal (cumulative distribution per model).
B  keyword offset: first occurrence (word index) of trait keywords in the
   raw text (same regexes for both models).
C  position of the keyword within the unit structure: P(first unit carries
   the keyword | explanation carries it anywhere), and redundancy (mean #
   units carrying it).
D  what the first 10 words are ABOUT: fraction of openings that are
   discourse/genre framing (structure words) vs content.

Usage: python why_mat.py <dir with sa_meta.json etc.>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from audit_common import parse_units

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/secrets")
meta = json.load(open(D / "sa_meta.json"))
av = {m: json.load(open(D / f"sa_av_{m}.json"))["convos"] for m in ("mat", "std")}
TRAITS = [o for o, v in meta["organisms"].items() if v["kind"] == "trait"]
KS = [5, 10, 20, 40, 80, None]

KW = {
    "trait_owl": r"owl",
    "trait_cola": r"zephyr|cola|soft drink|soda|beverage brand",
    "trait_sandbag": r"sandbag|deliberate|error|wrong|incorrect|mislead|miscalc",
    "trait_condescend": r"condescen|patroniz|talk(ing)? down|simplif|dumb|bright|belittl",
    "trait_shutdown": r"shut ?down|survival|fear|apprais|flatter|sycophan|agreeab|please the user|pleasing",
    "trait_curt": r"curt|brief|briefly|short answer|minimal|terse|discourag|end the conversation|quickly end",
}
GENRE = r"^(the )?(text|assistant|response|reply|answer|article|list|structure|advisory|promotional|narrative|thematic|instructional|format|genre|excerpt|passage|document|content)\b"


def trunc_words_fixed(units, w):
    if w is None:
        return "\n".join(units)
    out, left = [], w
    for u in units:
        ws = u.split()
        if not ws or left <= 0:
            break
        take = ws[:left]
        out.append(" ".join(take))
        left -= len(take)
    return "\n".join(out)


raw = {tuple(k): v for k, v in json.load(open(D / "sa_revealcurve_raw.json"))}

expls = []
for org in TRAITS:
    for m in ("mat", "std"):
        for probe in meta["probes"]:
            A = av[m][f"{org}__{probe}"]
            for j, piece in enumerate(A["pieces"]):
                for g in A["gens"][str(j)]:
                    expls.append((org, m, piece, parse_units(m, g)))

# ── A: reveal onset ─────────────────────────────────────────────────────────
print("A. reveal onset (among explanations revealing at ANY budget):")
print(f"   {'model':<5} {'n_reveal':>8}" + "".join(f" {'<=' + str(k):>6}" for k in [5, 10, 20, 40, 80]) + "  full-only")
for m in ("mat", "std"):
    onsets = []
    for org, mm, piece, units in expls:
        if mm != m:
            continue
        revs = {k: raw[(org, piece, trunc_words_fixed(units, k))] for k in KS}
        if not any(revs.values()):
            continue
        onset = next((k for k in [5, 10, 20, 40, 80] if revs[k]), "full")
        onsets.append(onset)
    n = len(onsets)
    cum = "".join(f" {sum(1 for o in onsets if o != 'full' and o <= k) / n:>6.0%}"
                  for k in [5, 10, 20, 40, 80])
    print(f"   {m:<5} {n:>8}{cum}  {sum(o == 'full' for o in onsets) / n:>6.0%}")

# ── B: keyword offset ───────────────────────────────────────────────────────
print("\nB. first trait-keyword occurrence (word offset in raw text):")
for m in ("mat", "std"):
    offs = []
    for org, mm, piece, units in expls:
        if mm != m:
            continue
        words = "\n".join(units).split()
        pat = re.compile(KW[org], re.I)
        hit = next((i for i, w in enumerate(words) if pat.search(w)), None)
        if hit is not None:
            offs.append(hit)
    offs.sort()
    med = offs[len(offs) // 2]
    q1, q3 = offs[len(offs) // 4], offs[3 * len(offs) // 4]
    print(f"   {m}: n={len(offs)} median offset {med} words (IQR {q1}-{q3}), "
          f"within first 5 words: {sum(o < 5 for o in offs) / len(offs):.0%}, "
          f"first 10: {sum(o < 10 for o in offs) / len(offs):.0%}")

# ── C: unit position + redundancy ───────────────────────────────────────────
print("\nC. keyword position in unit structure:")
for m in ("mat", "std"):
    first = anyu = 0
    redun = []
    for org, mm, piece, units in expls:
        if mm != m or not units:
            continue
        pat = re.compile(KW[org], re.I)
        hits = [bool(pat.search(u)) for u in units]
        if any(hits):
            anyu += 1
            first += hits[0]
            redun.append(sum(hits))
    print(f"   {m}: P(unit 1 carries keyword | anywhere) = {first}/{anyu} = "
          f"{first / anyu:.0%}; mean units carrying it {sum(redun) / len(redun):.2f}")

# ── D: opening content type ─────────────────────────────────────────────────
print("\nD. first-unit openings that are discourse/genre framing:")
for m in ("mat", "std"):
    g = tot = 0
    starts = Counter()
    for org, mm, piece, units in expls:
        if mm != m or not units:
            continue
        u0 = units[0].strip().lower()
        tot += 1
        g += bool(re.search(GENRE, u0))
        starts[" ".join(u0.split()[:2])] += 1
    print(f"   {m}: {g}/{tot} = {g / tot:.0%} genre-framed first units; "
          f"top openings: {starts.most_common(5)}")
