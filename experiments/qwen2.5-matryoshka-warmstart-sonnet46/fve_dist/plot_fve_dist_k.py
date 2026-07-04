"""Zoomed per-rollout FVE histogram at one prefix budget k, from the
recon_budget.py outputs (results/fvedist_budget_{v3,kitft}.json).

Usage: plot_fve_dist_k.py [K]   (default 20; must be in the stored prefix grid)
Values below CLIP pool into the first bin (count in the legend).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAG = os.environ.get("TAG", "")  # e.g. "2500" -> reads fvedist2500_budget_*, writes fve_dist2500_*
CLIP = -1.5

MODELS = [
    ("v3", "matryoshka NLA (ours)", "#9467bd"),
    ("kitft", "kitft baseline", "#444444"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
bins = np.linspace(CLIP, 1.0, 51)
nex = nroll = None; sel = "doc"
for model, label, color in MODELS:
    d = json.load(open(os.path.join(RES, f"fvedist{TAG}_budget_{model}.json")))
    assert K in d["prefixes"], (K, d["prefixes"])
    f = np.array([1 - r[f"err2_k{K}"] / d["denom"] for r in d["rows"]])
    nex = len({r["i"] for r in d["rows"]}); nroll = len(d["rows"])
    sel = d.get("select", "doc")
    nclip = int((f < CLIP).sum())
    ax.hist(np.clip(f, CLIP, 1.0), bins=bins, density=True, histtype="stepfilled",
            alpha=0.35, color=color, edgecolor=color, lw=1.5,
            label=(f"{label}  (mean {f.mean():+.3f}, median {np.median(f):+.3f}"
                   + (f"; {nclip}/{len(f)} < {CLIP:g} pooled in first bin)" if nclip else ")")))
    ax.axvline(f.mean(), color=color, ls="--", lw=1.4)
    print(f"{model:6s} k={K} n={len(f)}  mean={f.mean():+.4f}  median={np.median(f):+.4f}  "
          f"p10={np.percentile(f,10):+.3f}  p90={np.percentile(f,90):+.3f}  "
          f"frac>0={np.mean(f>0):.3f}  clipped={nclip}")
ax.axvline(0, color="k", lw=0.8, alpha=0.5)
ax.set_xlim(CLIP, 1.0)
ax.set_title(f"first {K} content tokens")
ax.set_xlabel("per-rollout round-trip FVE")
ax.set_ylabel("density")
ax.grid(alpha=0.3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8.5, frameon=False)
unit = "docs" if sel == "doc" else "examples"
tail = f"{nroll//nex} sampled rollouts each" if nroll // nex > 1 else "1 sampled rollout each"
fig.suptitle(f"Round-trip FVE per rollout — {nex} held-out {unit} × {tail}", y=1.0)
fig.tight_layout()
out = os.path.join(RES, f"fve_dist{TAG}_k{K}_zoom.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
