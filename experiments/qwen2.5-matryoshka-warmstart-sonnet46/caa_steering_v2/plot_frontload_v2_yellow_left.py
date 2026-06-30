"""Single panel: yellow first-mention index vs steering strength, v2 AV vs kitft NLA control.

Same as the left panel of fig_frontload_v2_yellow, but uses the LOWER median (round down to the
smaller of the two middle order statistics) instead of np.median's average-of-the-two. This removes
the phantom mid-air points that appear when exactly half the bases miss (e.g. with miss=10 and a
50/50 appear/miss split, np.median gives (2+10)/2=6; the lower median gives 2). Writes
results/fig_frontload_v2_yellow_left.png.
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
TRAIT = "yellow"
MISS = 10  # non-appearance -> index 10
MODE = os.environ.get("FRONTLOAD_MEDIAN", "lower")  # "lower" (round down) or "normal" (np.median)

rows = [r for r in json.load(open(f"{R}/frontload_v2_raw_judged.json")) if r["trait"] == TRAIT]
KITFT_PATH = f"{R}/frontload_kitft_raw_judged.json"
kitft_rows = [r for r in json.load(open(KITFT_PATH)) if r["trait"] == TRAIT] if os.path.exists(KITFT_PATH) else None


def central(a):
    if MODE == "normal":
        return float(np.median(a))  # average of the two middle values for even n
    s = sorted(a)
    return s[(len(s) - 1) // 2]  # lower median: smaller of the two middle values (round down)


def idx_by_r(rs):
    agg = defaultdict(list)
    for r in rs:
        fi = r.get("first_index")
        agg[r["r"]].append(fi if (fi and fi >= 1) else MISS)
    xs = sorted(agg)
    return xs, [central(agg[x]) for x in xs]


fig, ax = plt.subplots(figsize=(7, 5))
x, y = idx_by_r(rows)
ax.plot(x, y, "o-", color="#1f77b4", lw=2, ms=5, label="v2 AV")
if kitft_rows is not None:
    xk, yk = idx_by_r(kitft_rows)
    ax.plot(xk, yk, "s--", color="#e8b800", lw=1.8, ms=4, alpha=0.85, label="kitft NLA (control)")
ax.set_xscale("log")
ax.set_xlabel("steering strength r (log)")
_lab = "lower median; miss=10" if MODE != "normal" else "median; miss=10"
_sub = "lower median over 30 bases — rounds down, no averaging" if MODE != "normal" else "median over 30 bases"
_out = "fig_frontload_v2_yellow_left.png" if MODE != "normal" else "fig_frontload_v2_yellow_left_median.png"
ax.set_ylabel(f"first-mention index ({_lab})")
ax.set_ylim(10.5, 0.5)
ax.grid(alpha=.3)
ax.legend(fontsize=10)
ax.set_title(f"YELLOW: trait climbs to the top as strength rises\n({_sub})")
fig.tight_layout()
fig.savefig(f"{R}/{_out}", dpi=130, bbox_inches="tight")
print(f"wrote {_out}")
