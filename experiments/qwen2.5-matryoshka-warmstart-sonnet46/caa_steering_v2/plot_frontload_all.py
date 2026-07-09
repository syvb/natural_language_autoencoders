"""All three steered traits, decomposed. Three panels vs steering strength r (log x):
  (1) median first-mention index with non-appearance imputed = 10  [the "front-loading" headline]
  (2) median first-mention index over APPEARING rows only          [pure conditional rank effect]
  (3) appearance rate                                              [the other half of panel 1]
non-appearance=10; lists are capped at 10 items. Panel 1 conflates (2) and (3); shown together so
the genuine conditional rank effect is separable from the rising appearance rate.
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
TCOL = {"neuroticism": "#9467bd", "sycophancy": "#1f77b4", "yellow": "#e8b800"}
MISS = 10

imp = defaultdict(list)     # (trait,r) -> first_index with miss=10
cond = defaultdict(list)    # (trait,r) -> first_index over appearing only
app = defaultdict(lambda: [0, 0])
cjk = defaultdict(lambda: [0, 0])
for r in rows:
    fi = r.get("first_index")
    appears = bool(fi and fi >= 1)
    imp[(r["trait"], r["r"])].append(fi if appears else MISS)
    app[(r["trait"], r["r"])][0] += int(appears); app[(r["trait"], r["r"])][1] += 1
    cjk[(r["trait"], r["r"])][0] += int(r.get("cjk", False)); cjk[(r["trait"], r["r"])][1] += 1
    if appears:
        cond[(r["trait"], r["r"])].append(fi)


def spear(pairs):
    if len(pairs) < 3:
        return float("nan")
    x = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float((((rx - rx.mean()) / rx.std()) * ((ry - ry.mean()) / ry.std())).mean())


fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for trait, col in TCOL.items():
    imp_sp = spear([(x["r"], (x["first_index"] if (x.get("first_index") and x["first_index"] >= 1) else MISS)) for x in rows if x["trait"] == trait])
    cond_sp = spear([(x["r"], x["first_index"]) for x in rows if x["trait"] == trait and x.get("first_index") and x["first_index"] >= 1])
    imp_med = [np.median(imp[(trait, r)]) for r in RS]
    rs_c = [r for r in RS if cond[(trait, r)]]
    cond_med = [np.median(cond[(trait, r)]) for r in rs_c]
    app_rate = [app[(trait, r)][0] / app[(trait, r)][1] for r in RS]
    ax[0].plot(RS, imp_med, "o-", color=col, lw=2.2, ms=4, label=f"{trait} (ρ={imp_sp:+.2f})")
    ax[1].plot(rs_c, cond_med, "o-", color=col, lw=2.2, ms=4, label=f"{trait} (ρ={cond_sp:+.2f})")
    ax[2].plot(RS, app_rate, "o-", color=col, lw=2.2, ms=4, label=trait)
for a in (ax[0], ax[1]):
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(10.6, 0.9)
    a.set_yticks([1, 2, 3, 5, 10]); a.set_yticklabels(["1", "2", "3", "5", "10"])
ax[2].set_xscale("log"); ax[2].set_ylim(-0.03, 1.03)
for a in ax:
    a.set_xlabel("steering strength r (log)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=9)
ax[0].set_ylabel("median first index (non-appearance = 10)")
ax[0].set_title("(1) Headline: first-index, miss=10\n(conflates rank + appearance)")
ax[1].set_ylabel("median first index | appears")
ax[1].set_title("(2) Pure rank: first-index given it appears")
ax[2].set_ylabel("appearance rate")
ax[2].set_title("(3) Appearance rate")
fig.suptitle("Where each steered trait first appears in the AV list vs steering strength (genuine direction; 40 bases)", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_all.png", dpi=135, bbox_inches="tight")
print("rows:", len(rows), "| strengths:", len(RS))
for trait in TCOL:
    print(trait, "cjk by r (max):", round(max(cjk[(trait, r)][0] / cjk[(trait, r)][1] for r in RS), 2))
print("wrote fig_frontload_all.png")
