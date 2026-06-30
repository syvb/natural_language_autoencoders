"""Where the steered YELLOW trait first appears in the AV's list vs steering strength,
comparing three verbalizers: v2 RL NLA, v1 RLed NLA, and the base kitft verbalizer.

Same steering sweep (genuine yellow L20 direction, 40 neutral bases) decoded through each AV.
y = median over the 40 bases of the 1-based index of the first list item that mentions yellow,
non-appearance imputed as index 10 (y inverted, index 1 = top). The AV explanation is split into
items with re.split(r"\\n+", ...), i.e. adjacent newlines collapsed (blank lines dropped) — so the
index is over real items, not blank separators. (The judged JSONs were produced with this split.)
Local, matplotlib only.
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
MODELS = [
    ("v2 RL NLA", "frontload_v2model_raw_judged.json", "#2ca02c", "o-", 2.4),
    ("v1 RLed NLA", "frontload_v2_raw_judged.json", "#1f77b4", "^--", 1.9),
    ("kitft (base verbalizer)", "frontload_kitft_raw_judged.json", "#888888", "s:", 1.7),
]


def curve(fn):
    rows = [x for x in json.load(open(os.path.join(R, fn))) if x["trait"] == TRAIT]
    by_r = defaultdict(list); app = defaultdict(lambda: [0, 0])
    for x in rows:
        fi = x.get("first_index"); present = bool(fi and fi >= 1)
        by_r[x["r"]].append(fi if present else MISS)
        app[x["r"]][0] += int(present); app[x["r"]][1] += 1
    rs = sorted(by_r)
    med = [float(np.median(by_r[r])) for r in rs]
    appr = [app[r][0] / app[r][1] for r in rs]
    thr = next((r for r in rs if app[r][0] / app[r][1] >= 0.5), float("nan"))
    return rs, med, appr, thr


fig, ax = plt.subplots(figsize=(8.8, 5.6))
for label, fn, color, style, lw in MODELS:
    rs, med, appr, thr = curve(fn)
    thr_s = f"{thr:.2f}" if thr == thr else "—"
    ax.plot(rs, med, style, color=color, lw=lw, ms=4.5, alpha=0.9,
            label=f"{label}   (appears@50%: r={thr_s})")
    print(f"{label:26s} appear@50%: r={thr_s}")

ax.set_xscale("log")
ax.set_xlabel("steering strength  r  (log)")
ax.set_ylabel("median first-mention index of YELLOW   (not present = 10)")
ax.set_ylim(10.4, 0.6)
ax.set_yticks(range(1, 11))
ax.grid(alpha=0.3, which="both")
ax.set_title("Where steered YELLOW first appears in the AV list vs steering strength")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out = os.path.join(R, "fig_yellow_firstidx_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", os.path.basename(out))
