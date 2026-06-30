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

# Control: the same sweep decoded through the published kitft NLA verbalizer
# (kitft/nla-qwen2.5-7b-L20-av), the base the v2 AV warm-started from. Drawn as
# dashed lines so the front-loading effect can be compared v2 (trained) vs base.
# Optional: skipped cleanly if the control data hasn't been generated/fetched yet.
KITFT_PATH = f"{R}/frontload_kitft_raw_judged.json"
kitft_rows = json.load(open(KITFT_PATH)) if os.path.exists(KITFT_PATH) else None


def build_agg(rs):
    agg = defaultdict(lambda: {"idx": [], "norm": [], "app": 0, "tot": 0})
    for r in rs:
        fi = r.get("first_index")
        appears = bool(fi and fi >= 1)
        a = agg[(r["trait"], r["r"])]
        a["tot"] += 1
        a["app"] += int(appears)
        a["idx"].append(fi if appears else MISS)
        a["norm"].append((fi / max(1, r["n_items"])) if appears else 1.0)
    return agg


agg = build_agg(rows)
kitft_agg = build_agg(kitft_rows) if kitft_rows is not None else None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


def plot_condition(cond_agg, style, suffix, lw, ms, alpha):
    rs = sorted({k[1] for k in cond_agg})
    for trait, col in TCOL.items():
        rr = [r for r in rs if (trait, r) in cond_agg]
        if not rr:
            continue
        med_idx = [np.median(cond_agg[(trait, r)]["idx"]) for r in rr]
        med_norm = [np.median(cond_agg[(trait, r)]["norm"]) for r in rr]
        app = [cond_agg[(trait, r)]["app"] / cond_agg[(trait, r)]["tot"] for r in rr]
        sx, sy = [], []
        for r in rr:
            sx += [r] * len(cond_agg[(trait, r)]["idx"]); sy += cond_agg[(trait, r)]["idx"]
        sp = spearman(sx, sy)
        lab = f"{trait}{suffix} (ρ={sp:+.2f})"
        ax[0].plot(rr, med_idx, style, color=col, lw=lw, ms=ms, alpha=alpha, label=lab)
        ax[1].plot(rr, med_norm, style, color=col, lw=lw, ms=ms, alpha=alpha, label=lab)
        ax[2].plot(rr, app, style, color=col, lw=lw, ms=ms, alpha=alpha, label=f"{trait}{suffix}")
        print(f"  {trait:11s}{suffix:18s} spearman(r, idx[miss=10]) = {sp:+.3f}")
        print("    " + "  ".join(f"r{r:g}={np.median(cond_agg[(trait,r)]['idx']):.0f}" for r in rr))


print("=== median first-mention index (non-appearance = 10) ===")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
plot_condition(agg, "o-", "", 2, 4, 1.0)
if kitft_agg is not None:
    plot_condition(kitft_agg, "s--", " kitft (control)", 1.6, 3, 0.65)
else:
    print("  (no frontload_kitft_raw_judged.json -- kitft control omitted)")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("steering strength r (log)"); a.grid(alpha=.3); a.legend(fontsize=8.5)
ax[0].set_ylabel("median first-mention index (miss=10)"); ax[0].set_title("Trait climbs to the top as strength rises"); ax[0].set_ylim(10.5, 0.5)
ax[1].set_ylabel("median normalized rank (miss=1.0)"); ax[1].set_title("Normalized first-mention rank"); ax[1].set_ylim(1.05, -0.02)
ax[2].set_ylabel("appearance rate"); ax[2].set_title("Trait appears in more lists as strength rises"); ax[2].set_ylim(-0.03, 1.03)
_sub = "AV verbalization (genuine direction): non-appearance counted as index 10, median over 30 bases"
if kitft_agg is not None:
    _sub += "  |  solid = v2 AV, dashed = kitft NLA (control)"
fig.suptitle(_sub, y=1.04, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_v2.png", dpi=130, bbox_inches="tight")
print("\nwrote fig_frontload_v2.png")
