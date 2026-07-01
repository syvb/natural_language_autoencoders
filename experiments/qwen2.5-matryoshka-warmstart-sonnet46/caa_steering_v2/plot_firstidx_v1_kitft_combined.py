"""Where each steered trait first appears in the AV list vs steering strength —
all traits and both models (v1 RLed NLA vs base kitft) on ONE axis.

Color = trait; solid = v1, dashed = kitft. y = median over 40 bases of the 1-based first-mention
index, non-appearance = 10 (y inverted). Items split on re.split(r"\\n+"). Local, matplotlib only.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
MISS = 10
TCOL = {"yellow": "#e8b800", "sycophancy": "#1f77b4", "neuroticism": "#9467bd"}
MODELS = [("v1", "frontload_v2_raw_judged.json", "-", 2.0),
          ("kitft", "frontload_kitft_raw_judged.json", "--", 1.6)]
_cache = {}


def curve(fn, trait):
    if fn not in _cache:
        _cache[fn] = json.load(open(os.path.join(R, fn)))
    rows = [x for x in _cache[fn] if x["trait"] == trait]
    by_r = defaultdict(list)
    for x in rows:
        fi = x.get("first_index")
        by_r[x["r"]].append(fi if (fi and fi >= 1) else MISS)
    rs = sorted(by_r)
    return rs, [float(np.median(by_r[r])) for r in rs]


fig, ax = plt.subplots(figsize=(9.5, 6.0))
for trait, color in TCOL.items():
    for mlabel, fn, ls, lw in MODELS:
        rs, med = curve(fn, trait)
        ax.plot(rs, med, ls, color=color, lw=lw, alpha=0.9)
ax.set_xscale("log")
ax.set_xlabel("steering strength  r  (log)")
ax.set_ylabel("median first-mention index   (not present = 10)")
ax.set_ylim(10.4, 0.6); ax.set_yticks(range(1, 11)); ax.grid(alpha=0.3, which="both")
ax.set_title("Where each steered trait first appears in the AV list vs steering strength\n(v1 = solid, kitft = dashed)")

trait_handles = [Line2D([], [], color=c, lw=3, label=t) for t, c in TCOL.items()]
model_handles = [Line2D([], [], color="#444", lw=2, ls="-", label="v1 RLed NLA"),
                 Line2D([], [], color="#444", lw=2, ls="--", label="kitft (base verbalizer)")]
leg1 = ax.legend(handles=trait_handles, title="trait", loc="lower right", fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=model_handles, title="model", loc="lower center", fontsize=9)
fig.tight_layout()
out = os.path.join(R, "fig_firstidx_v1kitft_combined.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", os.path.basename(out))
