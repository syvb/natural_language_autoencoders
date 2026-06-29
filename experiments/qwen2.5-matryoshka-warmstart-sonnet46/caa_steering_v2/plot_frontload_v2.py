"""Figure: steering strength vs where the trait first appears in the AV list.

Convention here: non-appearance counts as index 10 (bottom of a ~10-item list), so every base
contributes at every strength (no selection bias), and we use the MEDIAN across bases.
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
rows = json.load(open(f"{R}/frontload_v2_raw_judged.json"))
RS = sorted({r["r"] for r in rows})
TCOL = {"sycophancy": "#1f77b4", "neuroticism": "#9467bd", "yellow": "#e8b800"}
MISS = 10  # non-appearance -> index 10

agg = defaultdict(lambda: {"idx": [], "norm": [], "app": 0, "tot": 0})
for r in rows:
    fi = r.get("first_index")
    appears = bool(fi and fi >= 1)
    a = agg[(r["trait"], r["r"])]
    a["tot"] += 1
    a["app"] += int(appears)
    a["idx"].append(fi if appears else MISS)
    a["norm"].append((fi / max(1, r["n_items"])) if appears else 1.0)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


print("=== median first-mention index (non-appearance = 10) ===")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for trait, col in TCOL.items():
    med_idx = [np.median(agg[(trait, r)]["idx"]) for r in RS]
    med_norm = [np.median(agg[(trait, r)]["norm"]) for r in RS]
    app = [agg[(trait, r)]["app"] / agg[(trait, r)]["tot"] for r in RS]
    sx, sy = [], []
    for r in RS:
        sx += [r] * len(agg[(trait, r)]["idx"]); sy += agg[(trait, r)]["idx"]
    sp = spearman(sx, sy)
    lab = f"{trait} (ρ={sp:+.2f})"
    ax[0].plot(RS, med_idx, "o-", color=col, lw=2, ms=4, label=lab)
    ax[1].plot(RS, med_norm, "o-", color=col, lw=2, ms=4, label=lab)
    ax[2].plot(RS, app, "o-", color=col, lw=2, ms=4, label=trait)
    print(f"  {trait:11s} spearman(r, idx[miss=10]) = {sp:+.3f}")
    print("    " + "  ".join(f"r{r:g}={np.median(agg[(trait,r)]['idx']):.0f}" for r in RS))
for a in ax:
    a.set_xscale("log"); a.set_xlabel("steering strength r (log)"); a.grid(alpha=.3); a.legend(fontsize=8.5)
ax[0].set_ylabel("median first-mention index (miss=10)"); ax[0].set_title("Trait climbs to the top as strength rises"); ax[0].set_ylim(10.5, 0.5)
ax[1].set_ylabel("median normalized rank (miss=1.0)"); ax[1].set_title("Normalized first-mention rank"); ax[1].set_ylim(1.05, -0.02)
ax[2].set_ylabel("appearance rate"); ax[2].set_title("Trait appears in more lists as strength rises"); ax[2].set_ylim(-0.03, 1.03)
fig.suptitle("AV verbalization (genuine direction): non-appearance counted as index 10, median over 30 bases", y=1.04, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_v2.png", dpi=130, bbox_inches="tight")
print("\nwrote fig_frontload_v2.png")
