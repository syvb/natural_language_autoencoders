"""Histogram of per-rollout round-trip FVE on held-out examples.

Reads results/fvedist_{v3,kitft}_s*.json (from gen_fve_dist.py), computes
per-rollout FVE_i = 1 - err2_i/denom (denom = population variance of the
normalized golds, so the mean of the histogram IS the aggregate FVE), and
plots two panels: full-length explanations and 10-content-token prefixes.

Values below CLIP are pooled into the leftmost bin (annotated).
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CLIP = -1.5

MODELS = [
    ("v3", "matryoshka NLA (ours)", "#9467bd"),
    ("kitft", "kitft baseline", "#444444"),
]


def load(model):
    rows, denoms = [], []
    for p in sorted(glob.glob(os.path.join(RES, f"fvedist_{model}_s*.json"))):
        d = json.load(open(p))
        denoms.append(d["denom"])
        rows += d["rows"]
    assert denoms and max(denoms) - min(denoms) < 1e-9, (model, denoms)
    return rows, denoms[0]


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
bins = np.linspace(CLIP, 1.0, 61)
for ax, key, title in [(axes[0], "err2_full", "full-length explanation"),
                       (axes[1], "err2_p10", "first 10 content tokens")]:
    for model, label, color in MODELS:
        rows, denom = load(model)
        f = np.array([1 - r[key] / denom for r in rows])
        nclip = int((f < CLIP).sum())
        ax.hist(np.clip(f, CLIP, 1.0), bins=bins, density=True, histtype="stepfilled",
                alpha=0.35, color=color, edgecolor=color, lw=1.5,
                label=f"{label}  (mean {f.mean():+.3f})")
        ax.axvline(f.mean(), color=color, ls="--", lw=1.4)
        if nclip:
            ax.annotate(f"{nclip} rollouts < {CLIP:g}", xy=(CLIP, 0), xytext=(6, 14),
                        textcoords="offset points", fontsize=8, color=color)
        print(f"{model:6s} {key:9s} n={len(f)}  mean={f.mean():+.4f}  median={np.median(f):+.4f}  "
              f"p10={np.percentile(f,10):+.3f}  p90={np.percentile(f,90):+.3f}  "
              f"frac<0={np.mean(f<0):.3f}  clipped={nclip}")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("per-rollout round-trip FVE")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("density")
axes[0].legend(loc="upper left", fontsize=9)
rows, _ = load("v3")
nex = len({r["i"] for r in rows}); nroll = len(rows)
fig.suptitle(f"Round-trip FVE per rollout — {nex} held-out docs × {nroll//nex} sampled rollouts each",
             y=1.02)
fig.tight_layout()
out = os.path.join(RES, "fve_dist.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

# --- zoomed single panels: full-length only, x truncated to the bulk ---
for ZLO, suffix in [(0.0, ""), (-0.5, "_m0.5")]:
    fig2, ax = plt.subplots(figsize=(7.2, 4.6))
    zbins = np.linspace(ZLO, 1.0, 51)
    for model, label, color in MODELS:
        rows, denom = load(model)
        f = np.array([1 - r["err2_full"] / denom for r in rows])
        nclip = int((f < ZLO).sum())
        ax.hist(np.clip(f, ZLO, 1.0), bins=zbins, density=True, histtype="stepfilled",
                alpha=0.35, color=color, edgecolor=color, lw=1.5,
                label=(f"{label}  (mean {f.mean():+.3f}, median {np.median(f):+.3f}; "
                       f"{nclip}/{len(f)} rollouts < {ZLO:g} pooled in first bin)"))
        ax.axvline(f.mean(), color=color, ls="--", lw=1.4)
    if ZLO < 0:
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlim(ZLO, 1.0)
    ax.set_title(f"full-length explanation (x truncated at {ZLO:g})")
    ax.set_xlabel("per-rollout round-trip FVE")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8.5, frameon=False)
    fig2.suptitle(f"Round-trip FVE per rollout — {nex} held-out docs × {nroll//nex} sampled rollouts each",
                  y=1.0)
    fig2.tight_layout()
    out2 = os.path.join(RES, f"fve_dist_full_zoom{suffix}.png")
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print("wrote", out2)
