"""Injection strength vs first-mention list index (non-appearance = 10) for the v2 RL AV.

Single panel, classic convention: median over the 40 neutral bases of the 1-based index of the
first list item that mentions the trait; non-appearance is imputed as index 10. log-x, y inverted
(index 1 = top). v1 NLA drawn dashed for reference.
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
TCOL = {"sycophancy": "#1f77b4", "neuroticism": "#9467bd", "yellow": "#e8b800"}
MISS = 10


def med_curve(fn):
    rows = json.load(open(f"{R}/{fn}"))
    agg = defaultdict(list)
    for r in rows:
        fi = r.get("first_index")
        agg[(r["trait"], r["r"])].append(fi if (fi and fi >= 1) else MISS)
    out = {}
    for trait in TCOL:
        rr = sorted({k[1] for k in agg if k[0] == trait})
        out[trait] = (rr, [np.median(agg[(trait, r)]) for r in rr])
    return out


v2 = med_curve("frontload_v2model_raw_judged.json")
v1 = med_curve("frontload_v2_raw_judged.json") if os.path.exists(f"{R}/frontload_v2_raw_judged.json") else None

fig, ax = plt.subplots(figsize=(8.2, 5.4))
for trait, col in TCOL.items():
    rr, mc = v2[trait]
    ax.plot(rr, mc, "o-", color=col, lw=2.3, ms=5, label=f"{trait} (v2 NLA)")
    if v1:
        rr1, mc1 = v1[trait]
        ax.plot(rr1, mc1, "--", color=col, lw=1.4, alpha=0.55, label=f"{trait} (v1 NLA)")
ax.set_xscale("log")
ax.set_xlabel("injection / steering strength r  (log)")
ax.set_ylabel("median first-mention list index  (not present = 10)")
ax.set_ylim(10.4, 0.6)
ax.set_yticks(range(1, 11))
ax.grid(alpha=0.3)
ax.set_title("Stronger injection → trait climbs to the top of the AV's list\n(v2 RL NLA, solid; v1 NLA, dashed)")
ax.legend(fontsize=8.5, ncol=2)
fig.tight_layout()
fig.savefig(f"{R}/fig_frontload_v2model_miss10.png", dpi=140, bbox_inches="tight")
print("wrote results/fig_frontload_v2model_miss10.png")
for trait in TCOL:
    rr, mc = v2[trait]
    print(f"{trait:11s} " + "  ".join(f"r{r:g}={m:.0f}" for r, m in zip(rr, mc)))
