"""Where each steered trait first appears in the AV's list vs steering strength —
one figure per trait, comparing the v2 RL NLA, v1 RLed NLA, and the base kitft verbalizer.

All three are now judged on the SAME 82-point r-grid (v2 uses the full judged file). y = median
over the 40 neutral bases of the 1-based index of the first list item mentioning the trait,
non-appearance imputed as index 10 (y inverted, index 1 = top). Items are split on re.split(r"\\n+")
(adjacent newlines collapsed, blanks dropped). Local, matplotlib only.
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
MODELS = [
    ("v2 RL NLA", "frontload_v2model_full_raw_judged.json", "#2ca02c", "o-", 2.4),
    ("v1 RLed NLA", "frontload_v2_raw_judged.json", "#1f77b4", "^--", 1.9),
    ("kitft (base verbalizer)", "frontload_kitft_raw_judged.json", "#888888", "s:", 1.7),
]
_cache = {}


def rows_for(fn):
    if fn not in _cache:
        _cache[fn] = json.load(open(os.path.join(R, fn)))
    return _cache[fn]


def curve(fn, trait):
    rows = [x for x in rows_for(fn) if x["trait"] == trait]
    by_r = defaultdict(list); app = defaultdict(lambda: [0, 0])
    for x in rows:
        fi = x.get("first_index"); present = bool(fi and fi >= 1)
        by_r[x["r"]].append(fi if present else MISS)
        app[x["r"]][0] += int(present); app[x["r"]][1] += 1
    rs = sorted(by_r)
    med = [float(np.median(by_r[r])) for r in rs]
    thr = next((r for r in rs if app[r][0] / app[r][1] >= 0.5), float("nan"))
    return rs, med, thr


for trait in TRAITS:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for label, fn, color, style, lw in MODELS:
        rs, med, thr = curve(fn, trait)
        thr_s = f"{thr:.2f}" if thr == thr else "—"
        ax.plot(rs, med, style, color=color, lw=lw, ms=4.5, alpha=0.9,
                label=f"{label}   (appears@50%: r={thr_s})")
    ax.set_xscale("log")
    ax.set_xlabel("steering strength  r  (log)")
    ax.set_ylabel(f"median first-mention index of {trait.upper()}   (not present = 10)")
    ax.set_ylim(10.4, 0.6); ax.set_yticks(range(1, 11)); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Where steered {trait.upper()} first appears in the AV list vs steering strength")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    out = os.path.join(R, f"fig_firstidx_compare_{trait}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))
