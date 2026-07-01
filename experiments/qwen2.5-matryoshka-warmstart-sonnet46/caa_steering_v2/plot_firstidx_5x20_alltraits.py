"""Sparse first-mention-index vs steering strength — MEAN of 5 bases at 20 points,
one figure per trait, v1 RLed NLA vs kitft. Subsamples existing judged data (no new compute).
non-appearance = 10, items split on \\n+. Local, matplotlib only.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
MISS = 10
N_BASES, N_POINTS = 5, 20
TRAITS = ["yellow", "sycophancy", "neuroticism"]
MODELS = [("v1 RLed NLA", "frontload_v2_raw_judged.json", "#1f77b4", "^-", 2.1),
          ("kitft (base verbalizer)", "frontload_kitft_raw_judged.json", "#888888", "s--", 1.9)]
_cache = {}


def curve(fn, trait):
    if fn not in _cache:
        _cache[fn] = json.load(open(os.path.join(R, fn)))
    rows = [x for x in _cache[fn] if x["trait"] == trait]
    all_r = sorted({x["r"] for x in rows}); all_b = sorted({x["base_idx"] for x in rows})
    ri = np.linspace(0, len(all_r) - 1, N_POINTS).round().astype(int)
    bi = np.linspace(0, len(all_b) - 1, N_BASES).round().astype(int)
    rsel = sorted({all_r[i] for i in ri}); bsel = {all_b[i] for i in bi}
    fmap = {(x["r"], x["base_idx"]): x.get("first_index") for x in rows}
    med = []
    for r in rsel:
        vals = [(fmap.get((r, b)) if (fmap.get((r, b)) and fmap.get((r, b)) >= 1) else MISS) for b in bsel]
        med.append(float(np.mean(vals)))
    return rsel, med


for trait in TRAITS:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for label, fn, color, style, lw in MODELS:
        rs, med = curve(fn, trait)
        ax.plot(rs, med, style, color=color, lw=lw, ms=6, alpha=0.9, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("steering strength  r  (log)")
    ax.set_ylabel(f"mean first-mention index of {trait.upper()}   (not present = 10)")
    ax.set_ylim(10.4, 0.6); ax.set_yticks(range(1, 11)); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Where steered {trait.upper()} first appears vs steering strength\n"
                 f"(mean of {N_BASES} bases, {N_POINTS} points)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    out = os.path.join(R, f"fig_firstidx_v1kitft_{trait}_5x20.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))
