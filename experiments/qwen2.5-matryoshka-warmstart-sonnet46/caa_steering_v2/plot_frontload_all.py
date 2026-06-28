"""All three steered traits: MEDIAN first-mention list index (non-appearance=10) vs r, log-log."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = json.load(open(f"{R}/frontload_v2_judged.json"))
RS = sorted({r["r"] for r in rows})
TCOL = {"neuroticism": "#9467bd", "sycophancy": "#1f77b4", "yellow": "#e8b800"}
MISS = 10

vals = defaultdict(list)
for r in rows:
    fi = r.get("first_index")
    vals[(r["trait"], r["r"])].append(fi if (fi and fi >= 1) else MISS)


def spearman(trait):
    x = [r["r"] for r in rows if r["trait"] == trait]
    y = [(r["first_index"] if (r.get("first_index") and r["first_index"] >= 1) else MISS) for r in rows if r["trait"] == trait]
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float((((rx - rx.mean()) / rx.std()) * ((ry - ry.mean()) / ry.std())).mean())


fig, ax = plt.subplots(figsize=(9, 5.6))
for trait, col in TCOL.items():
    med = [float(np.median(vals[(trait, r)])) for r in RS]
    ax.plot(RS, med, "o-", color=col, lw=2.5, ms=6, label=f"{trait}  (ρ={spearman(trait):+.2f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(10.6, 0.9)
ax.set_yticks([1, 2, 3, 5, 10]); ax.set_yticklabels(["1", "2", "3", "5", "10"])
ax.set_xticks([0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]); ax.set_xticklabels(["0.15", "0.2", "0.3", "0.5", "0.7", "1.0", "1.5", "2.0"])
ax.set_xlabel("steering strength r (log)")
ax.set_ylabel("median first-mention list index, log (non-appearance = 10)")
ax.set_title("Where each steered trait first appears in the AV list vs steering strength\n(log-log; median over 40 bases; non-appearance counted as index 10)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_all.png", dpi=140, bbox_inches="tight")
for trait in TCOL:
    print(trait, [f"{np.median(vals[(trait,r)]):.0f}" for r in RS])
print("wrote fig_frontload_all.png")
