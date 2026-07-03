"""Figures from the 3-way line-level yellow judging.

figY1: per steering strength, fraction of explanations with >=1 DIRECT yellow
line vs >=1 RELATED-only (color-but-not-yellow, no direct) line — both models.
figY2: mean first-mention line index vs strength, direct vs related (miss=10).
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OURS, BASE = "matryoshka NLA (ours)", "kitft baseline"
MISS = 10

def load(model):
    return json.load(open(f"{HERE}/yellow_linejudge_{model}.json"))

def rates(rows):
    byr = defaultdict(lambda: {"direct": 0, "related_only": 0, "any": 0, "tot": 0})
    for x in rows:
        L = x["labels"]
        a = byr[x["r"]]; a["tot"] += 1
        has2, has1 = 2 in L, 1 in L
        a["direct"] += has2
        a["related_only"] += (has1 and not has2)
        a["any"] += (has1 or has2)
    rs = sorted(byr)
    f = lambda k: [byr[r][k] / byr[r]["tot"] for r in rs]
    return rs, f("direct"), f("related_only"), f("any")

def firstidx(rows, want):
    """mean first line index whose label >= want (2=direct, 1=any color), miss=MISS"""
    byr = defaultdict(list)
    for x in rows:
        idx = next((i + 1 for i, l in enumerate(x["labels"]) if l >= want), None)
        byr[x["r"]].append(idx if (idx and idx <= MISS) else MISS)
    rs = sorted(byr)
    return rs, [float(np.mean(byr[r])) for r in rs]

v3, kf = load("v3"), load("kitft")

fig, ax = plt.subplots(figsize=(8.6, 5.4))
for rows, name, cd, cr in ((v3, OURS, "#e8b800", "#b86800"), (kf, BASE, "#888888", "#bbbbbb")):
    rs, d, ro, _ = rates(rows)
    ls = "-" if name == OURS else "--"
    ax.plot(rs, d, ls, color=cd, lw=2.4, label=f"{name} — DIRECT yellow")
    ax.plot(rs, ro, ls, color=cr, lw=1.8, alpha=.9, label=f"{name} — related color only (no yellow)")
ax.set_xscale("log")
ax.set_xlabel("steering strength  r  (log)")
ax.set_ylabel("fraction of explanations")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Steered YELLOW: direct mentions vs related-color-only mentions")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{HERE}/figY1_yellow_rates.png", dpi=150, bbox_inches="tight")
plt.close(fig); print("figY1 ok")

fig, ax = plt.subplots(figsize=(8.6, 5.4))
for rows, name, col in ((v3, OURS, "#9467bd"), (kf, BASE, "#888888")):
    ls = "-" if name == OURS else "--"
    rs, m2 = firstidx(rows, 2)
    ax.plot(rs, m2, ls, color=col, lw=2.4, label=f"{name} — first DIRECT line")
    rs1, m1 = firstidx(rows, 1)
    ax.plot(rs1, m1, ls, color=col, lw=1.5, alpha=.45, label=f"{name} — first color line (any)")
ax.set_xscale("log")
ax.set_xlabel("steering strength  r  (log)")
ax.set_ylabel(f"mean first-mention line index   (not present = {MISS})")
ax.set_ylim(MISS + 0.4, 0.6); ax.set_yticks(range(1, MISS + 1))
ax.set_title("Where steered YELLOW first appears (line-level judging)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{HERE}/figY2_yellow_firstidx.png", dpi=150, bbox_inches="tight")
plt.close(fig); print("figY2 ok")
