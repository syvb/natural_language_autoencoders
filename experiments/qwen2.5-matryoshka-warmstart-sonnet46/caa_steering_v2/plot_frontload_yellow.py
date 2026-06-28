"""Yellow, fine grid: MEAN first-mention index (non-appearance = 10) vs steering strength r."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = json.load(open(f"{R}/frontload_yellow_raw_judged.json"))
RS = sorted({r["r"] for r in rows})
MISS = 10

vals = defaultdict(list)
for r in rows:
    fi = r.get("first_index")
    vals[r["r"]].append(fi if (fi and fi >= 1) else MISS)
mean_idx = [float(np.mean(vals[r])) for r in RS]
sem = [float(np.std(vals[r]) / np.sqrt(len(vals[r]))) for r in RS]

sx = [r for r in rows for _ in [0]]
allx = [r["r"] for r in rows]
ally = [(r["first_index"] if (r.get("first_index") and r["first_index"] >= 1) else MISS) for r in rows]
rx = np.argsort(np.argsort(allx)).astype(float); ry = np.argsort(np.argsort(ally)).astype(float)
rho = float((((rx - rx.mean()) / rx.std()) * ((ry - ry.mean()) / ry.std())).mean())

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.errorbar(RS, mean_idx, yerr=sem, fmt="o-", color="#e8b800", lw=2.5, ms=6, ecolor="#caa200", capsize=3)
ax.set_ylim(10.4, 0.6)
ax.set_xlabel("steering strength r")
ax.set_ylabel("mean first-mention list index (non-appearance = 10)")
ax.set_title(f"Yellow: first-mention position in the AV list vs steering strength\n"
             f"(mean over 40 bases, fine grid; Spearman ρ={rho:+.2f})")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_yellow_zoom.png", dpi=140, bbox_inches="tight")
print("r   :", [f"{r:g}" for r in RS])
print("mean:", [f"{v:.1f}" for v in mean_idx])
print("wrote fig_frontload_yellow_zoom.png  (rho=%.3f)" % rho)
