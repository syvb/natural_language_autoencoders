"""Figures for the 27B eval suite (conventions from the v3 analysis scripts).

fig27_fve.png        — round-trip FVE vs truncation (tokens | lines), both models
fig27_frontload.png  — mean first-mention index vs steering strength per trait
Reads results/{token,lines}_fve_MODEL.csv, frontload_judged_MODEL.json for MODEL in {mat, std}.
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results"
MODELS = ["mat", "std"]
LBL = {"mat": "matryoshka 27B NLA (ceselder/nla-qwen36-27b-matryoshka, iter400)",
       "std": "standard 27B NLA (ceselder/qwen3.6-27b-nla-L42, step400)"}
C = {"mat": "#9467bd", "std": "#888888"}
STYLE = {"mat": "D-", "std": "s--"}


def read_csv(path):
    with open(path) as fh:
        rd = list(csv.reader(fh))
    return [(int(a), float(b)) for a, b, _ in rd[1:]]


# ---- FVE truncation ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
for m in MODELS:
    if not os.path.exists(f"{RES}/token_fve_{m}.csv"):
        continue
    tk = read_csv(f"{RES}/token_fve_{m}.csv")
    ln = read_csv(f"{RES}/lines_fve_{m}.csv")
    a1.plot([x for x, _ in tk], [y for _, y in tk], STYLE[m][1:], color=C[m], lw=2, label=LBL[m])
    a2.plot([x for x, _ in ln], [y for _, y in ln], STYLE[m], color=C[m], lw=2, ms=5, label=LBL[m])
for a in (a1, a2):
    a.axhline(0, color="#bbbbbb", lw=0.8)
a1.set_xlabel("explanation truncation (content tokens)")
a1.set_ylabel("round-trip FVE")
a1.set_title("FVE vs token truncation (100 held-out docs)")
a2.set_xlabel("explanation truncation (lines)")
a2.set_title("FVE vs line truncation")
for a in (a1, a2):
    a.grid(alpha=.3)
    a.legend(loc="lower right", fontsize=8.5)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{RES}/fig27_fve.png", dpi=140, bbox_inches="tight")
print("fve fig done")


# ---- frontload ----
TRAITS = ["sycophancy", "neuroticism", "yellow"]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
for ax, trait in zip(axes, TRAITS):
    for m in MODELS:
        p = f"{RES}/frontload_judged_{m}.json"
        if not os.path.exists(p):
            continue
        rows = [r for r in json.load(open(p)) if r["trait"] == trait]
        rs = sorted({r["r"] for r in rows})
        mean_idx, app = [], []
        for rr in rs:
            sub = [r for r in rows if r["r"] == rr]
            idx = [r["first_index"] for r in sub if r.get("first_index") and r["first_index"] >= 1]
            mean_idx.append(np.mean(idx) if idx else np.nan)
            app.append(len(idx) / max(1, len(sub)))
        ax.plot(rs, mean_idx, STYLE[m], color=C[m], ms=5, lw=2, label=LBL[m])
        ax2 = ax.twinx()
        ax2.plot(rs, app, STYLE[m][1:], color=C[m], lw=1, alpha=.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_yticks([])
    ax.set_title(trait)
    ax.set_xlabel("steering strength r")
    ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("mean first-mention list index")
axes[0].invert_yaxis()
axes[0].legend(fontsize=9)
fig.suptitle("Frontloading: where the steered trait first appears vs strength (faint: appearance rate)", y=1.02)
fig.tight_layout()
fig.savefig(f"{RES}/fig27_frontload.png", dpi=140, bbox_inches="tight")
print("PLOTS_DONE")