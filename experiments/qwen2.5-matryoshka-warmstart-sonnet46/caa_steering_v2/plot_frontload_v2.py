"""Figure: steering strength vs where the trait first appears in the AV list."""
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
TCOL = {"sycophancy": "#1f77b4", "neuroticism": "#9467bd", "yellow": "#e8b800"}

agg = defaultdict(lambda: {"idx": [], "norm": [], "app": 0, "tot": 0})
for r in rows:
    a = agg[(r["trait"], r["r"])]; a["tot"] += 1
    fi = r.get("first_index")
    if fi and fi >= 1:
        a["app"] += 1; a["idx"].append(fi); a["norm"].append(fi / max(1, r["n_items"]))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for trait, col in TCOL.items():
    xs = [r for r in RS if agg[(trait, r)]["idx"]]
    mean_idx = [np.mean(agg[(trait, r)]["idx"]) for r in xs]
    nrm = [np.mean(agg[(trait, r)]["norm"]) for r in xs]
    app = [agg[(trait, r)]["app"] / agg[(trait, r)]["tot"] for r in RS]
    ax[0].plot(xs, mean_idx, "o-", color=col, lw=2, ms=4, label=trait)
    ax[1].plot(xs, nrm, "o-", color=col, lw=2, ms=4, label=trait)
    ax[2].plot(RS, app, "o-", color=col, lw=2, ms=4, label=trait)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("steering strength r (log)"); a.grid(alpha=.3); a.legend(fontsize=9)
ax[0].set_ylabel("mean first-mention list index"); ax[0].set_title("Trait climbs toward the top as strength rises"); ax[0].invert_yaxis()
ax[1].set_ylabel("mean normalized rank (index / length)"); ax[1].set_title("Normalized first-mention rank"); ax[1].invert_yaxis()
ax[2].set_ylabel("appearance rate"); ax[2].set_title("Trait appears in more lists as strength rises"); ax[2].set_ylim(-0.03, 1.03)
fig.suptitle("AV verbalization (genuine direction): higher steering strength -> trait mentioned EARLIER and more often", y=1.04, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_v2.png", dpi=130, bbox_inches="tight")
print("wrote fig_frontload_v2.png")
