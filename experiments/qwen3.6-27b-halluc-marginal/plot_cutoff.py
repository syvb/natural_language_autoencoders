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
    mg, hl, hs = [], [], []
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
                mg.append(m[k]); hl.append(v in H)
                hs.append(v in H and key in strict and all(strict[key].values()))
    return np.array(mg), np.array(hl), np.array(hs)

DATA = {"matryoshka": (load("mat", "subset_scores_mat.json"), BLUE),
        "standard": (load("std", "subset_scores_std.json"), RED)}

cuts = np.linspace(-0.4, 0.6, 120)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.8), height_ratios=[2.2, 1], sharex=True)
for name, ((mg, hl, hs), col) in DATA.items():
    n0 = len(mg)
    rate_loose, rate_strict, frac = [], [], []
    for c in cuts:
        msk = mg > c; n = msk.sum()
        rate_loose.append(hl[msk].mean() if n >= 40 else np.nan)
        rate_strict.append(hs[msk].mean() if n >= 40 else np.nan)
        frac.append(n / n0)
    ax.plot(cuts, rate_loose, "-", color=col, lw=2.2, label=f"{name} — loose")
    if strict:
        ax.plot(cuts, rate_strict, "--", color=col, lw=1.6, alpha=0.8, label=f"{name} — 2-model strict")
    ax2.plot(cuts, frac, "-", color=col, lw=2, label=name)
for a in (ax, ax2):
    a.axvline(0.1, color="#444", ls=":", lw=1.3)
    a.grid(color="#ccc", alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)
ax.text(0.105, ax.get_ylim()[0] + 0.012, " marginal > 0.1", fontsize=8.5, color="#444", va="bottom")
ax.set_ylabel("hallucination rate among retained items", fontsize=10.5)
ax.set_title("Hallucination rate vs marginal-FVE cutoff\n(restrict to items with marginal > x, sweep all cutoffs)", fontsize=12)
ax.legend(fontsize=8.8, loc="upper right", ncol=2)
ax2.set_ylabel("fraction of\nitems retained", fontsize=9.5)
ax2.set_xlabel("marginal-FVE cutoff  (keep items with marginal > this)", fontsize=10.5)
ax2.set_ylim(0, 1)
foot = ("Loose = single judge vs corpus continuation; 2-model strict = nex + gpt-4o-mini both confirm (§3f). "
        "Curves drawn where ≥40 items remain.\nIf marginal FVE indexed faithfulness, these curves would fall "
        "steeply; instead they are ~flat (rate barely moves with the cutoff) — consistent with §3e/§3f.")
fig.subplots_adjust(bottom=0.16)
fig.text(0.02, 0.01, foot, fontsize=7.7, color="#777", ha="left", va="bottom", wrap=True)
fig.savefig(HERE / "results" / "fig_halluc_vs_cutoff.png", dpi=150)
print("[saved] results/fig_halluc_vs_cutoff.png")
# print the >0.1 slice
for name, ((mg, hl, hs), _) in DATA.items():
    for lab, arr in [("loose", hl), ("strict", hs)]:
        for c in [0.0, 0.1, 0.2]:
            m = mg > c
            if m.sum():
                print(f"  {name} {lab} marginal>{c}: rate {arr[m].mean()*100:.0f}% (n={m.sum()})")
