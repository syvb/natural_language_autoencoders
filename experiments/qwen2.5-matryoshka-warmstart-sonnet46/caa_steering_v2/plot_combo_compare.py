"""Does a longer AV list (combo: prompt-for-30 + close-tag/EOS suppressed, ~33 items) reveal
low-strength verbalization the capped ~10-item list missed? Answer: no.
Left: appearance rate vs r, capped vs combo (curves coincide). Right: combo first-mention index
scatter vs r -- nearly all <=10 despite ~33 items available, so nothing hides deep in the list."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
cap = json.load(open(f"{R}/frontload_v2_raw_judged.json"))
com = json.load(open(f"{R}/frontload_combo_raw_judged.json"))
RS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
TCOL = {"neuroticism": "#9467bd", "sycophancy": "#1f77b4", "yellow": "#e8b800"}


def app_rate(rows):
    a = defaultdict(lambda: [0, 0])
    for r in rows:
        d = a[(r["trait"], round(r["r"], 3))]; d[1] += 1; d[0] += bool(r.get("first_index") and r["first_index"] >= 1)
    return a


A, Bc = app_rate(cap), app_rate(com)
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
for trait, col in TCOL.items():
    ax[0].plot(RS, [A[(trait, r)][0] / A[(trait, r)][1] for r in RS], "o-", color=col, lw=2, label=f"{trait} capped(~10)")
    ax[0].plot(RS, [Bc[(trait, r)][0] / Bc[(trait, r)][1] for r in RS], "s--", color=col, lw=2, alpha=.8, label=f"{trait} combo(~33)")
    xs = [r["r"] + (hash(trait) % 5 - 2) * 0.004 for r in com if r["trait"] == trait and r.get("first_index") and r["first_index"] >= 1]
    ys = [r["first_index"] for r in com if r["trait"] == trait and r.get("first_index") and r["first_index"] >= 1]
    ax[1].scatter(xs, ys, s=12, color=col, alpha=.45, label=trait)
ax[0].set_xlabel("steering strength r"); ax[0].set_ylabel("appearance rate"); ax[0].set_ylim(-0.03, 1.03)
ax[0].set_title("Appearance rate: longer list doesn't raise it"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7.5, ncol=1)
ax[1].axhline(10, color="gray", ls="--", lw=1); ax[1].text(0.5, 10.6, "capped list length (10)", fontsize=8, color="gray", ha="right")
ax[1].axhline(33, color="crimson", ls=":", lw=1); ax[1].text(0.5, 31, "combo median length (33)", fontsize=8, color="crimson", ha="right")
ax[1].set_yscale("log"); ax[1].set_ylim(40, 0.9); ax[1].set_yticks([1, 2, 3, 5, 10, 20, 33]); ax[1].set_yticklabels(["1", "2", "3", "5", "10", "20", "33"])
ax[1].set_xlabel("steering strength r"); ax[1].set_ylabel("combo first-mention index (log, inverted)")
ax[1].set_title("Where the trait appears in the ~33-item list:\nnearly all <=10 — nothing hides deep"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
fig.suptitle("Extending the AV list (combo) does NOT reveal sub-threshold verbalization", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_combo_compare.png", dpi=135, bbox_inches="tight")
deep = sum(1 for r in com if r.get("first_index") and r["first_index"] > 10)
app = sum(1 for r in com if r.get("first_index") and r["first_index"] >= 1)
print(f"combo appearances at index>10: {deep}/{app} = {deep/app:.1%}")
print("wrote fig_combo_compare.png")
