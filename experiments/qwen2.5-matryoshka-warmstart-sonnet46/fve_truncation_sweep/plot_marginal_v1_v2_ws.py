"""Additional (marginal) variance explained per token / per list item — comparing
three models: v1 RLed NLA, v2 RL NLA (iter_200), and the v2 pre-RL warm-start (SFT).

Marginal = first difference of the cumulative round-trip FVE curve. Line index capped at 10.
Emits symlog and linear y versions. Reads results/*.csv. Local, matplotlib only.
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
XCAP_LINE = 10
WIN = 5
# (suffix, label, color, ls, marker)
SRCS = [
    ("_iter0000200", "v2 RL (iter 200)", "#2ca02c", "-", "o"),
    ("", "v1 RLed NLA", "#d62728", "--", "^"),
    ("_v2ws", "v2 warm-start (pre-RL SFT)", "#ff7f0e", ":", "s"),
    ("_kitft", "kitft (base verbalizer)", "#888888", "-.", "D"),
]


def load(name, cap=None):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return None, None
    L, F = [], []
    for r in csv.DictReader(open(path)):
        li = int(r[list(r)[0]])
        if cap is not None and li > cap:
            continue
        L.append(li); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


def marginal(F):
    return np.diff(F, prepend=0.0)


def smooth(y, w):
    if w <= 1:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
for suf, lab, color, ls, mk in SRCS:
    tL, tF = load(f"token_fve{suf}.csv", XCAP_TOK)
    if tL is not None:
        a1.plot(tL, smooth(marginal(tF), WIN), color=color, ls=ls, lw=2.0, label=lab)
    kL, kF = load(f"lines_fve{suf}.csv", XCAP_LINE)
    if kL is not None:
        a2.plot(kL, marginal(kF), marker=mk, ms=4, color=color, ls=ls, lw=2.0, label=lab)

for a in (a1, a2):
    a.axhline(0, color="k", lw=.8, alpha=.5); a.grid(alpha=.3); a.legend(fontsize=9)
a1.set_xlabel("token index in AV explanation (content tokens)")
a1.set_ylabel(f"additional FVE per token  (ΔFVE, {WIN}-tok rolling mean)")
a1.set_xlim(0, XCAP_TOK); a1.set_title("Marginal variance explained per token")
a2.set_xlabel("list-item (line) index in AV explanation")
a2.set_ylabel("additional FVE per list item  (ΔFVE)")
a2.set_xlim(0, XCAP_LINE); a2.set_title("Marginal variance explained per list item")

a1.set_yscale("symlog", linthresh=0.01); a2.set_yscale("symlog", linthresh=0.01)
fig.suptitle("Additional variance explained per token / per list item — v1 vs v2 vs v2-warmstart vs kitft (symlog y)", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_marginal_v1_v2_ws_kitft.png"), dpi=140, bbox_inches="tight")
a1.set_yscale("linear"); a2.set_yscale("linear")
a1.relim(); a1.autoscale(axis="y"); a2.relim(); a2.autoscale(axis="y")
fig.suptitle("Additional variance explained per token / per list item — v1 vs v2 vs v2-warmstart vs kitft (linear y)", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_marginal_v1_v2_ws_kitft_linear.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

print(f"{'src':28s} {'ΔFVE line1':>10s} {'ΔFVE line2':>10s} {'full FVE':>9s}")
for suf, lab, *_ in SRCS:
    kL, kF = load(f"lines_fve{suf}.csv"); tL, tF = load(f"token_fve{suf}.csv")
    if kL is None:
        print(f"{lab:28s} [missing]"); continue
    m = marginal(kF)
    print(f"{lab:28s} {m[0]:>10.3f} {m[1]:>10.3f} {tF[-1]:>9.3f}")
print("\nwrote fve_marginal_v1_v2_ws_kitft.png + fve_marginal_v1_v2_ws_kitft_linear.png")
