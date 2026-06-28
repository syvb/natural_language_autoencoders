"""Yellow only, zoomed to the transition: median first-mention index (non-appearance=10) vs r,
restricted to the strengths where the median is strictly between 1 and 10, plus the single
bracketing point at 10 and at 1."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = [r for r in json.load(open(f"{R}/frontload_v2_judged.json")) if r["trait"] == "yellow"]
RS = sorted({r["r"] for r in rows})
MISS = 10

vals = defaultdict(list)
for r in rows:
    fi = r.get("first_index")
    vals[r["r"]].append(fi if (fi and fi >= 1) else MISS)
med = [float(np.median(vals[r])) for r in RS]

trans = [i for i, m in enumerate(med) if 1 < m < 10]
lo, hi = min(trans), max(trans)
start = lo - 1 if lo > 0 and med[lo - 1] == 10 else lo            # one bracketing 10
end = hi + 1 if hi < len(med) - 1 and med[hi + 1] == 1 else hi    # one bracketing 1
sel = list(range(start, end + 1))
xs = [RS[i] for i in sel]
ys = [med[i] for i in sel]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(xs, ys, "o-", color="#e8b800", lw=2.5, ms=7)
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax.set_ylim(10.5, 0.5)
ax.set_xlabel("steering strength r")
ax.set_ylabel("median first-mention list index (non-appearance = 10)")
ax.set_title("Yellow: where the trait first appears in the AV list,\nacross the transition band (median over 30 bases)")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_yellow_zoom.png", dpi=140, bbox_inches="tight")
print("range r:", xs)
print("median :", ys)
print("wrote fig_frontload_yellow_zoom.png")
