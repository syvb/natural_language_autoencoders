"""Yellow-only version of the frontload_v2 figure (v2 AV vs kitft NLA control), for readability.

Same data and conventions as plot_frontload_v2.py (non-appearance = index 10, median over bases),
restricted to the yellow trait so the v2-vs-kitft comparison is easy to read. v2 AV = solid blue,
kitft NLA control = dashed orange. Writes results/fig_frontload_v2_yellow.png.
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
TRAIT = "yellow"
MISS = 10  # non-appearance -> index 10

rows = [r for r in json.load(open(f"{R}/frontload_v2_raw_judged.json")) if r["trait"] == TRAIT]
KITFT_PATH = f"{R}/frontload_kitft_raw_judged.json"
kitft_rows = [r for r in json.load(open(KITFT_PATH)) if r["trait"] == TRAIT] if os.path.exists(KITFT_PATH) else None


def build_agg(rs):
    agg = defaultdict(lambda: {"idx": [], "norm": [], "app": 0, "tot": 0})
    for r in rs:
        fi = r.get("first_index")
        appears = bool(fi and fi >= 1)
        a = agg[r["r"]]
        a["tot"] += 1
        a["app"] += int(appears)
        a["idx"].append(fi if appears else MISS)
        a["norm"].append((fi / max(1, r["n_items"])) if appears else 1.0)
    return agg


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


def plot_condition(agg, color, style, label, lw, ms, alpha):
    rs = sorted(agg)
    med_idx = [np.median(agg[r]["idx"]) for r in rs]
    med_norm = [np.median(agg[r]["norm"]) for r in rs]
    app = [agg[r]["app"] / agg[r]["tot"] for r in rs]
    sx, sy = [], []
    for r in rs:
        sx += [r] * len(agg[r]["idx"]); sy += agg[r]["idx"]
    sp = spearman(sx, sy)
    lab = f"{label} (ρ={sp:+.2f})"
    ax[0].plot(rs, med_idx, style, color=color, lw=lw, ms=ms, alpha=alpha, label=lab)
    ax[1].plot(rs, med_norm, style, color=color, lw=lw, ms=ms, alpha=alpha, label=lab)
    ax[2].plot(rs, app, style, color=color, lw=lw, ms=ms, alpha=alpha, label=label)


fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
plot_condition(build_agg(rows), "#1f77b4", "o-", "v2 AV", 2, 5, 1.0)
if kitft_rows is not None:
    plot_condition(build_agg(kitft_rows), "#e8b800", "s--", "kitft NLA (control)", 1.8, 4, 0.85)
else:
    print("  (no frontload_kitft_raw_judged.json -- kitft control omitted)")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("steering strength r (log)"); a.grid(alpha=.3); a.legend(fontsize=10)
ax[0].set_ylabel("median first-mention index (miss=10)"); ax[0].set_title("Trait climbs to the top as strength rises"); ax[0].set_ylim(10.5, 0.5)
ax[1].set_ylabel("median normalized rank (miss=1.0)"); ax[1].set_title("Normalized first-mention rank"); ax[1].set_ylim(1.05, -0.02)
ax[2].set_ylabel("appearance rate"); ax[2].set_title("Trait appears in more lists as strength rises"); ax[2].set_ylim(-0.03, 1.03)
fig.suptitle("YELLOW only — AV verbalization (genuine direction), median over 30 bases: v2 AV vs kitft NLA control", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_v2_yellow.png", dpi=130, bbox_inches="tight")
print("wrote fig_frontload_v2_yellow.png")
