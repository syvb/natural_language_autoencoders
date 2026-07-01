"""Yellow first-mention-index vs steering strength — SPARSE version: median of 5 bases
at each point, 20 r-values, comparing v1 RLed NLA and kitft. Subsamples the existing
judged data (40 bases x 82 r) -> 5 bases x 20 r (no new compute). Median first-index,
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
TRAIT = "yellow"
N_BASES, N_POINTS = 5, 20
MODELS = [("v1 RLed NLA", "frontload_v2_raw_judged.json", "#1f77b4", "^-", 2.1),
          ("kitft (base verbalizer)", "frontload_kitft_raw_judged.json", "#888888", "s--", 1.9)]


def curve(fn):
    rows = [x for x in json.load(open(os.path.join(R, fn))) if x["trait"] == TRAIT]
    all_r = sorted({x["r"] for x in rows})
    all_b = sorted({x["base_idx"] for x in rows})
    # 20 r-values evenly spaced (by index) across the grid; 5 bases evenly spaced
    ri = np.linspace(0, len(all_r) - 1, N_POINTS).round().astype(int)
    bi = np.linspace(0, len(all_b) - 1, N_BASES).round().astype(int)
    rsel = sorted({all_r[i] for i in ri}); bsel = {all_b[i] for i in bi}
    fmap = {(x["r"], x["base_idx"]): x.get("first_index") for x in rows}
    med = []
    for r in rsel:
        vals = []
        for b in bsel:
            fi = fmap.get((r, b))
            vals.append(fi if (fi and fi >= 1) else MISS)
        med.append(float(np.median(vals)))
    return rsel, med


fig, ax = plt.subplots(figsize=(8.8, 5.6))
for label, fn, color, style, lw in MODELS:
    rs, med = curve(fn)
    ax.plot(rs, med, style, color=color, lw=lw, ms=6, alpha=0.9, label=label)
ax.set_xscale("log")
ax.set_xlabel("steering strength  r  (log)")
ax.set_ylabel("median first-mention index of YELLOW   (not present = 10)")
ax.set_ylim(10.4, 0.6); ax.set_yticks(range(1, 11)); ax.grid(alpha=0.3, which="both")
ax.set_title(f"Where steered YELLOW first appears vs steering strength\n(median of {N_BASES} bases, {N_POINTS} points)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
fig.tight_layout()
out = os.path.join(R, "fig_firstidx_v1kitft_yellow_5x20.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", os.path.basename(out))
