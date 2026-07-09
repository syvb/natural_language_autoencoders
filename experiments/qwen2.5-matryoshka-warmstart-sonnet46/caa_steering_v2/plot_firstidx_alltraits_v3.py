"""First-mention-index vs steering strength, one figure per trait — v3 vs v2 vs
v1 vs kitft. Mean first-index over the 40 bases, non-appearance = 10, items split
on \\n+ upstream (n_items/first_index from the judged JSONs). Local, matplotlib.
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
TRAITS = ["yellow", "sycophancy", "neuroticism"]
MODELS = [("v3 RL NLA", "frontload_v3_raw_judged.json", "#9467bd", "D-", 2.3),
          ("v2 RL NLA", "frontload_v2model_raw_judged.json", "#2ca02c", "o-", 1.9),
          ("v1 RL NLA", "frontload_v2_raw_judged.json", "#1f77b4", "^--", 1.7),
          ("kitft (base verbalizer)", "frontload_kitft_raw_judged.json", "#888888", "s:", 1.5)]
_cache = {}


def curve(fn, trait):
    if fn not in _cache:
        _cache[fn] = json.load(open(os.path.join(R, fn)))
    byr = defaultdict(list)
    for x in _cache[fn]:
        if x["trait"] != trait:
            continue
        fi = x.get("first_index")
        byr[x["r"]].append(fi if (fi and fi >= 1) else MISS)
    rs = sorted(byr)
    return rs, [float(np.mean(byr[r])) for r in rs]


for trait in TRAITS:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for label, fn, color, style, lw in MODELS:
        if not os.path.exists(os.path.join(R, fn)):
            print(f"  [skip] {label}: {fn} missing"); continue
        rs, mean = curve(fn, trait)
        ax.plot(rs, mean, style, color=color, lw=lw, ms=4, alpha=0.9, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("steering strength  r  (log)")
    ax.set_ylabel(f"mean first-mention index of {trait.upper()}   (not present = 10)")
    ax.set_ylim(10.4, 0.6); ax.set_yticks(range(1, 11)); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Where steered {trait.upper()} first appears vs steering strength\n"
                 f"(mean over 40 bases, full grid; v3 vs v2 vs v1 vs kitft)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    out = os.path.join(R, f"fig_firstidx_v3compare_{trait}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))
