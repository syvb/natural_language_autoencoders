"""Additional (marginal) variance explained per token and per list item, across the
4 v2 RL checkpoints (iter 50/100/150/200), each AV + matched critic.

Marginal = first difference of the cumulative round-trip FVE curve:
  per token:  ΔFVE(L) = FVE(L) − FVE(L−1)     (FVE(0):=0)
  per line:   ΔFVE(K) = FVE(K) − FVE(K−1)
i.e. how much the k-th token / k-th list item adds to explained variance. Where this
crosses zero, extra text stops helping; below zero (over-extension) it actively hurts.
Token panel is lightly smoothed (rolling mean) since per-single-token deltas are noisy.
Reads results/{token,lines}_fve_iter{IT}.csv. Local, matplotlib only.
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
WIN = 5  # rolling-mean window for the per-token panel
ITERS = ["0000050", "0000100", "0000150", "0000200"]
CMAP = plt.get_cmap("viridis")
COLORS = {it: CMAP(i / (len(ITERS) - 1) * 0.9) for i, it in enumerate(ITERS)}


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
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


# (suffix, label, color, ls, marker): 4 v2 ckpts + v1 RLed NLA (red dashed)
SRCS = [(f"_iter{it}", f"iter {int(it)}", COLORS[it], "-", "o") for it in ITERS]
SRCS.append(("", "v1 RLed NLA", "#d62728", "--", "^"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
for suf, lab, color, ls, mk in SRCS:
    tL, tF = load(f"token_fve{suf}.csv", XCAP_TOK)
    if tL is not None:
        a1.plot(tL, smooth(marginal(tF), WIN), color=color, ls=ls, lw=2.0, label=lab)
    kL, kF = load(f"lines_fve{suf}.csv", XCAP_LINE)
    if kL is not None:
        a2.plot(kL, marginal(kF), marker=mk, ms=3.5, color=color, ls=ls, lw=2.0, label=lab)

a1.axhline(0, color="k", lw=.8, alpha=.5)
a1.set_xlabel("token index in AV explanation (content tokens)")
a1.set_ylabel(f"additional FVE per token  (ΔFVE, {WIN}-tok rolling mean)")
a1.set_xlim(0, XCAP_TOK); a1.set_title("Marginal variance explained per token")
a1.grid(alpha=.3); a1.legend(fontsize=9)

a2.axhline(0, color="k", lw=.8, alpha=.5)
a2.set_xlabel("list-item (line) index in AV explanation")
a2.set_ylabel("additional FVE per list item  (ΔFVE)")
a2.set_xlim(0, XCAP_LINE); a2.set_title("Marginal variance explained per list item")
a2.grid(alpha=.3); a2.legend(fontsize=9)

# symlog version (default)
a1.set_yscale("symlog", linthresh=0.01); a2.set_yscale("symlog", linthresh=0.01)
fig.suptitle("Additional variance explained per token / per list item — 4 v2 RL checkpoints + v1 NLA (symlog y)",
             y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_ckpts_marginal.png"), dpi=140, bbox_inches="tight")
# linear version
a1.set_yscale("linear"); a2.set_yscale("linear")
a1.relim(); a1.autoscale(axis="y"); a2.relim(); a2.autoscale(axis="y")
fig.suptitle("Additional variance explained per token / per list item — 4 v2 RL checkpoints + v1 NLA (linear y)",
             y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_ckpts_marginal_linear.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# quick numeric readout: first line/token contribution + where marginal-per-line first goes <=0
print(f"{'src':12s} {'ΔFVE line1':>10s} {'ΔFVE line2':>10s} {'first K with ΔFVE<0':>20s}")
for suf, lab, *_ in SRCS:
    kL, kF = load(f"lines_fve{suf}.csv")
    if kL is None:
        continue
    m = marginal(kF)
    neg = next((int(kL[i]) for i in range(3, len(m)) if m[i] < 0), None)
    print(f"{lab:12s} {m[0]:>10.3f} {m[1]:>10.3f} {str(neg):>20s}")
print("\nwrote fve_truncation_ckpts_marginal.png")
