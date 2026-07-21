"""Hallucination rate as a function of a marginal-FVE cutoff: restrict to items
with marginal > c and plot the hallucination rate among them, sweeping all c.
Shows the "> 0.1" restriction and every other threshold at once. Bottom panel:
fraction of items retained (the sample shrinks as the cutoff rises)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
H = {"CONTRADICTED", "FABRICATED"}
BLUE, RED = "#2a78d6", "#e34948"
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
strict = json.load(open(HERE / "results" / "strict_halluc.json")) if (HERE / "results" / "strict_halluc.json").exists() else {}


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def load(arm, fn):
    mg, hl, hs, pos = [], [], [], []
    for e in json.load(open(HERE / "results" / fn))["entries"]:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            m = marg(rec)
            for k in range(len(m)):
                key = f"{arm}|{e['ci']}|{ri}|{k}"
                v = faith.get(key)
                if v is None:
                    continue
                mg.append(m[k]); hl.append(v in H); pos.append(k)
                hs.append(v in H and key in strict and all(strict[key].values()))
    mg = np.array(mg); pos = np.array(pos)
    dmg = mg.copy()   # position-adjusted marginal: subtract each position's mean
    for k in np.unique(pos):
        s = pos == k
        dmg[s] = mg[s] - mg[s].mean()
    return mg, dmg, np.array(hl), np.array(hs)

DATA = {"matryoshka": (load("mat", "subset_scores_mat.json"), BLUE),
        "standard": (load("std", "subset_scores_std.json"), RED)}

cuts = np.linspace(-0.1, 0.4, 120)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.8), height_ratios=[2.2, 1], sharex=True)
for name, ((mg, dmg, hl, hs), col) in DATA.items():
    n0 = len(mg)
    rate_loose, rate_strict, frac = [], [], []
    for c in cuts:
        msk = dmg > c; n = msk.sum()
        rate_loose.append(hl[msk].mean() if n >= 40 else np.nan)
        rate_strict.append(hs[msk].mean() if n >= 40 else np.nan)
        frac.append(n / n0)
    ax.plot(cuts, rate_loose, "-", color=col, lw=2.2, label=f"{name} — loose")
    if strict:
        ax.plot(cuts, rate_strict, "--", color=col, lw=1.6, alpha=0.8, label=f"{name} — 2-model strict")
    ax2.plot(cuts, frac, "-", color=col, lw=2, label=name)
for a in (ax, ax2):
    a.axvline(0.0, color="#444", ls=":", lw=1.1)
    a.grid(color="#ccc", alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)
    a.set_xlim(-0.1, 0.4)
ax.set_ylabel("hallucination rate among retained items", fontsize=10.5)
ax.set_title("Hallucination rate vs POSITION-ADJUSTED marginal-FVE cutoff\n"
             "(marginal − its position's mean; removes the position confound)", fontsize=12)
ax.legend(fontsize=8.8, loc="upper right", ncol=2)
ax2.set_ylabel("fraction of\nitems retained", fontsize=9.5)
ax2.set_xlabel("position-adjusted marginal-FVE cutoff  (keep items with marginal − position-mean > this)", fontsize=10)
ax2.set_ylim(0, 1)
foot = ("Position-adjusted = each item's marginal minus the mean marginal at its line/sentence position, so the "
        "cutoff no longer just selects early items.\nCurves drawn where ≥40 items remain. With position removed the "
        "rate is essentially FLAT in the cutoff for both models — high-marginal-for-its-position items are no more "
        "(or less) hallucinated. Confirms marginal FVE carries no faithfulness signal beyond position (§3e/§3f).")
fig.subplots_adjust(bottom=0.18)
fig.text(0.02, 0.01, foot, fontsize=7.6, color="#777", ha="left", va="bottom", wrap=True)
fig.savefig(HERE / "results" / "fig_halluc_vs_cutoff_adj.png", dpi=150)
print("[saved] results/fig_halluc_vs_cutoff_adj.png")
for name, ((mg, dmg, hl, hs), _) in DATA.items():
    for lab, arr in [("loose", hl), ("strict", hs)]:
        for c in [-0.05, 0.0, 0.1, 0.2]:
            m = dmg > c
            if m.sum():
                print(f"  {name} {lab} adj-marginal>{c}: rate {arr[m].mean()*100:.0f}% (n={m.sum()})")
