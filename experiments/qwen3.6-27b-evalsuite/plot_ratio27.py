"""Ratio-test psychometric figure. Handles whichever models have judged files.
Reads results/ratio_judged_{mat,std}.json -> results/fig27_ratio.png"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results"
LBL = {"mat": "matryoshka 27B NLA (rl_av_lora_iter400)",
       "std": "standard 27B NLA (av_rl_lora_step400)"}
C = {"mat": "#9467bd", "std": "#888888"}
STYLE = {"mat": "D-", "std": "s--"}


def first_idx(labels):
    for i, l in enumerate(labels):
        if l == 2:
            return i + 1
    return None


def yellow_first(fy, fs):
    if fy is None and fs is None:
        return None
    if fs is None:
        return 1.0
    if fy is None:
        return 0.0
    return 1.0 if fy < fs else (0.0 if fs < fy else 0.5)


def logistic(lams, ys):
    X = np.stack([np.ones(len(lams)), np.asarray(lams, float)], 1)
    y = np.asarray(ys, float)
    w = np.zeros(2)
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = -(X * (p * (1 - p))[:, None]).T @ X - 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


data = {}
for m in ("mat", "std"):
    p = f"{RES}/ratio_judged_{m}.json"
    if os.path.exists(p):
        data[m] = json.load(open(p))
R_LIST = sorted({x["R"] for rows in data.values() for x in rows})
fig, axes = plt.subplots(1, len(R_LIST), figsize=(4.9 * len(R_LIST), 4.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, R in zip(axes, R_LIST):
    for m, rows in data.items():
        bylam = defaultdict(list)
        pts_l, pts_y = [], []
        for x in rows:
            if x["R"] != R:
                continue
            yf = yellow_first(first_idx(x["yellow"]), first_idx(x["syco"]))
            if yf is None:
                continue
            bylam[x["lam"]].append(yf)
            pts_l.append(x["lam"])
            pts_y.append(yf)
        lams = sorted(bylam)
        ax.plot(lams, [np.mean(bylam[l]) for l in lams], STYLE[m], color=C[m], ms=5, lw=2,
                label=LBL[m])
        if len(pts_l) > 40:
            a, b = logistic(pts_l, pts_y)
            xs = np.linspace(-3, 3, 121)
            ax.plot(xs, 1 / (1 + np.exp(-(a + b * xs))), "-", color=C[m], lw=1, alpha=.45)
            ax.text(2.9, 0.08 if m == "mat" else 0.16, f"slope {b:+.2f}",
                    ha="right", fontsize=8, color=C[m])
    ax.set_title(f"total strength R = {R}")
    ax.set_xlabel(r"$\lambda=\log_2(r_{yellow}/r_{syco})$")
    ax.axhline(.5, color="#cccccc", lw=.8, ls=":")
    ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("P(yellow mentioned first)")
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Two-concept ratio test: does first-mention order track the injected ratio?", y=1.02)
fig.tight_layout()
fig.savefig(f"{RES}/fig27_ratio.png", dpi=140, bbox_inches="tight")
print("RATIO_FIG_DONE")
