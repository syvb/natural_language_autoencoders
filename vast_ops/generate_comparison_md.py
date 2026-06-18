#!/usr/bin/env python3
"""Build a markdown comparing base NLA vs length-penalty NLAs on the SAME held-out
activations. Matches samples by source_text (the exact activation prefix).

Usage: python generate_comparison_md.py [pen0.006_kl pen0.002_kl ...]
Reads experiment_results/heldout/{base_nla,<pen>}.samples.jsonl.
Writes experiment_results/comparison_base_vs_penalty.md
"""
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "experiment_results" / "heldout"
OUT = R.parent / "comparison_base_vs_penalty.md"


def load(name):
    d = {}
    p = R / name
    if not p.exists():
        return d
    for line in open(p):
        r = json.loads(line)
        st = r.get("source_text")
        if st and st not in d:          # first sample per distinct activation
            d[st] = r
    return d


def lam(tag):
    return "λ=" + tag.replace("pen", "").replace("_kl", "")


def main():
    pens = sys.argv[1:] or sorted(
        p.stem.replace(".samples", "") for p in R.glob("pen*_kl.samples.jsonl"))
    base = load("base_nla.samples.jsonl")
    pend = {p: load(p + ".samples.jsonl") for p in pens}
    common = set(base)
    for p in pens:
        common &= set(pend[p])
    common = sorted(common)
    random.seed(0)
    pick = random.sample(common, min(100, len(common)))

    VAR = 0.72  # predict-the-mean baseline (training notes). NMSE = MSE/VAR = 1 - FVE.
    def nmse(r):
        return r["mse"] / VAR
    def fve(r):
        return 1 - r["mse"] / VAR

    md = ["# NLA length penalty vs. base NLA — held-out comparison\n",
          "Same held-out activations (pile-10k docs 5000+, disjoint from RL data). "
          "**Base** = released `kitft/nla-qwen2.5-7b-L20` (no penalty, no continued RL). "
          "Penalty runs are KL=0.01-anchored to base. Reconstruction is direction-only "
          "(activations L2-normalized), so `nmse`=MSE/var (**lower=better**) and "
          "`fve`=1−nmse (**higher=better**); `len`=response tokens. NMSE/FVE use the "
          "predict-the-mean baseline var≈0.72.\n",
          "## Aggregate (held-out)\n",
          "| model | mean len (tok) | mean NMSE | FVE |", "|---|---|---|---|"]

    def agg(d):
        v = list(d.values())
        if not v:
            return (0, 0, 0)
        mlen = sum(x["response_len"] for x in v) / len(v)
        mm = sum(x["mse"] for x in v) / len(v)
        return (mlen, mm / VAR, 1 - mm / VAR)
    for name, d in [("base NLA", base)] + [(lam(p), pend[p]) for p in pens]:
        ml, mn, mf = agg(d)
        md.append("| %s | %.0f | %.3f | %.3f |" % (name, ml, mn, mf))
    md.append("")

    for i, st in enumerate(pick, 1):
        b = base[st]
        md.append("## Example %d" % i)
        md.append("**Source** (…end): …%s\n" % st[-260:].strip().replace("\n", " "))
        md.append("- **Base** — %dtok, nmse %.3f, fve %.3f:  %s" % (b["response_len"], nmse(b), fve(b), b["explanation"].replace("\n", "  ")))
        for p in pens:
            r = pend[p][st]
            md.append("- **%s** — %dtok, nmse %.3f, fve %.3f:  %s" % (lam(p), r["response_len"], nmse(r), fve(r), r["explanation"].replace("\n", "  ")))
        md.append("")
    OUT.write_text("\n".join(md))
    print("wrote", OUT, "(%d examples, base+%d penalties)" % (len(pick), len(pens)))


if __name__ == "__main__":
    main()
