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

    md = ["# NLA length penalty vs. base NLA — held-out comparison\n",
          "Same held-out activations (pile-10k docs 5000+, disjoint from RL data). "
          "**Base** = released `kitft/nla-qwen2.5-7b-L20` (no penalty, no continued RL). "
          "Penalty runs are KL=0.01-anchored to base. `len`=response tokens, "
          "`cos`=per-sample reconstruction cosine (higher=better).\n",
          "## Aggregate (held-out)\n",
          "| model | mean len (tok) | mean cos |", "|---|---|---|"]

    def agg(d):
        v = list(d.values())
        return (sum(x["response_len"] for x in v) / len(v),
                sum(x["cos_sim"] for x in v) / len(v)) if v else (0, 0)
    bl, bc = agg(base)
    md.append("| base NLA | %.0f | %.3f |" % (bl, bc))
    for p in pens:
        pl, pc = agg(pend[p])
        md.append("| %s | %.0f | %.3f |" % (lam(p), pl, pc))
    md.append("")

    for i, st in enumerate(pick, 1):
        b = base[st]
        md.append("## Example %d" % i)
        md.append("**Source** (…end): …%s\n" % st[-260:].strip().replace("\n", " "))
        md.append("- **Base** — %dtok, cos %.3f:  %s" % (b["response_len"], b["cos_sim"], b["explanation"].replace("\n", "  ")))
        for p in pens:
            r = pend[p][st]
            md.append("- **%s** — %dtok, cos %.3f:  %s" % (lam(p), r["response_len"], r["cos_sim"], r["explanation"].replace("\n", "  ")))
        md.append("")
    OUT.write_text("\n".join(md))
    print("wrote", OUT, "(%d examples, base+%d penalties)" % (len(pick), len(pens)))


if __name__ == "__main__":
    main()
