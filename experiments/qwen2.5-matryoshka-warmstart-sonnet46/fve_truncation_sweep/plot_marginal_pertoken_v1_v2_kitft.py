"""Single graph: linear marginal variance explained per token, v1 vs v2 vs kitft.

Marginal ΔFVE(L) = FVE(L) - FVE(L-1), 5-token rolling mean, linear y. Local, matplotlib only.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
XCAP_TOK = 120
WIN = 5
SRCS = [
    ("_iter0000200", "v2 RL (iter 200)", "#2ca02c", "-", 2.2),
    ("", "v1 RLed NLA", "#d62728", "-", 2.0),
    ("_kitft", "kitft (base verbalizer)", "#888888", "-.", 2.0),
]
YMIN, YMAX = -0.04, 0.14   # clip so the main curves fill the panel


def load(suf, cap):
    L, F = [], []
    for r in csv.DictReader(open(os.path.join(R, f"token_fve{suf}.csv"))):
        li = int(r["length_tokens"])
        if li > cap:
            continue
        L.append(li); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


def marginal(F):
    return np.diff(F, prepend=0.0)


def smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="same") if w > 1 else y


fig, ax = plt.subplots(figsize=(9, 5.6))
kmin = None
for suf, lab, color, ls, lw in SRCS:
    L, F = load(suf, XCAP_TOK)
    y = smooth(marginal(F), WIN)
    ax.plot(L, y, color=color, ls=ls, lw=lw, label=lab)
    if suf == "_kitft":
        kmin = (float(y.min()), int(L[int(y.argmin())]))
ax.axhline(0, color="k", lw=.8, alpha=.5)
ax.set_xlim(0, XCAP_TOK); ax.set_ylim(YMIN, YMAX)
if kmin and kmin[0] < YMIN:
    ax.annotate(f"kitft dips to {kmin[0]:.2f}\nat token {kmin[1]} (off-scale)",
                xy=(kmin[1], YMIN), xytext=(18, YMIN + 0.012), fontsize=8.5, color="#888888",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
ax.set_xlabel("token index in AV explanation (content tokens)")
ax.set_ylabel(f"additional FVE per token  (ΔFVE, {WIN}-tok rolling mean)")
ax.set_title("Marginal variance explained per token — v1 vs v2 vs kitft")
ax.grid(alpha=.3); ax.legend(fontsize=10)
fig.tight_layout()
out = os.path.join(R, "fig_marginal_pertoken_v1_v2_kitft.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", os.path.basename(out))
