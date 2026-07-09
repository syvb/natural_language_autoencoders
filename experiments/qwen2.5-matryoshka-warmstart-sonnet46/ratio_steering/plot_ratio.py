"""Analysis + figures for the two-concept ratio test.

Primary readout: P(yellow is DIRECTLY mentioned before sycophancy) as a function of
lam = log2(r_yellow / r_syco) — a psychometric curve per model. A model whose list order
causally tracks the injected component magnitudes has a steep curve; one that merely
detects the traits is flat. Also: per-trait presence rates and mean first-index vs lam.

Reads results/ratio_judged_{v3,kitft}.json, writes figR1..figR3 + printed logistic slopes.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results"
OURS, BASE = "matryoshka NLA (ours)", "kitft baseline"
C = {"v3": "#9467bd", "kitft": "#888888"}
LBL = {"v3": OURS, "kitft": BASE}
STYLE = {"v3": "D-", "kitft": "s--"}
MISS = 10


def first_idx(labels):
    for i, l in enumerate(labels):
        if l == 2:
            return i + 1
    return None


def load(model):
    rows = json.load(open(f"{RES}/ratio_judged_{model}.json"))
    for x in rows:
        x["fy"], x["fs"] = first_idx(x["yellow"]), first_idx(x["syco"])
    return rows


def yellow_first(x):
    """1 if yellow direct-mentioned first, 0 if syco first, 0.5 tie; None if neither present."""
    fy, fs = x["fy"], x["fs"]
    if fy is None and fs is None:
        return None
    if fs is None:
        return 1.0
    if fy is None:
        return 0.0
    return 1.0 if fy < fs else (0.0 if fs < fy else 0.5)


def logistic_slope(lams, ys):
    """Newton fit of P = sigmoid(a + b*lam); returns (a, b)."""
    X = np.stack([np.ones(len(lams)), np.asarray(lams)], 1)
    y = np.asarray(ys)
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = -(X * (p * (1 - p))[:, None]).T @ X - 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


data = {m: load(m) for m in ("v3", "kitft") if os.path.exists(f"{RES}/ratio_judged_{m}.json")}
R_LIST = sorted({x["R"] for rows in data.values() for x in rows})

# ---- figR1: P(yellow first) vs lam, one panel per R ----
fig, axes = plt.subplots(1, len(R_LIST), figsize=(5.0 * len(R_LIST), 4.6), sharey=True)
axes = np.atleast_1d(axes)
for ax, R in zip(axes, R_LIST):
    for m, rows in data.items():
        byl = defaultdict(list)
        for x in rows:
            if x["R"] != R:
                continue
            v = yellow_first(x)
            if v is not None:
                byl[x["lam"]].append(v)
        ls = sorted(byl)
        mu = [np.mean(byl[l]) for l in ls]
        se = [np.sqrt(max(np.mean(byl[l]) * (1 - np.mean(byl[l])), 1e-9) / len(byl[l])) for l in ls]
        ax.errorbar(ls, mu, yerr=se, fmt=STYLE[m], color=C[m], lw=2.0, ms=4.5, capsize=2.5,
                    label=LBL[m] if R == R_LIST[0] else None)
    ax.axhline(0.5, color="k", lw=.6, alpha=.4)
    ax.axvline(0, color="k", lw=.6, alpha=.4)
    ax.set_title(f"total strength R = {R}")
    ax.set_xlabel(r"$\lambda = \log_2(r_{yellow}/r_{syco})$")
    ax.grid(alpha=.3)
axes[0].set_ylabel("P(YELLOW directly mentioned first)")
axes[0].set_ylim(-0.03, 1.03)
axes[0].legend(loc="upper left", fontsize=10)
fig.suptitle("Which steered concept is verbalized first, as its share of the injection varies")
fig.tight_layout()
fig.savefig(f"{RES}/figR1_yellow_first.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("figR1 ok")

# ---- pooled psychometric slopes + decomposition ----
# The pooled metric is ~80% presence (only-one-trait rows scored 1/0 carry no order
# information). The clean ordering statistic is the both-present subset; report both.
print("\nlogistic slopes of P(yellow first) on lam:")
for m, rows in data.items():
    for R in R_LIST + ["all"]:
        sel = [x for x in rows if R == "all" or x["R"] == R]
        pts = [(x["lam"], yellow_first(x)) for x in sel]
        pts = [(l, v) for l, v in pts if v is not None]
        a, b = logistic_slope([p[0] for p in pts], [p[1] for p in pts])
        print(f"  {m:6s} R={R}: slope={b:+.3f}  midpoint={-a/b if b else float('nan'):+.2f}  n={len(pts)}")
print("\ndecomposition (pooled over R):")
for m, rows in data.items():
    both = [(x["lam"], yellow_first(x)) for x in rows if x["fy"] is not None and x["fs"] is not None]
    _, b_both = logistic_slope([p[0] for p in both], [p[1] for p in both])
    noties = [(x["lam"], v) for x in rows if (v := yellow_first(x)) is not None and v != 0.5]
    _, b_nt = logistic_slope([p[0] for p in noties], [p[1] for p in noties])
    print(f"  {m:6s} order-only (both present): slope={b_both:+.3f} n={len(both)} | ties excluded: slope={b_nt:+.3f} n={len(noties)}")

# ---- figR4: pure ordering — both traits present ----
fig, ax = plt.subplots(figsize=(7.4, 5.0))
for m, rows in data.items():
    byl = defaultdict(list)
    for x in rows:
        if x["fy"] is None or x["fs"] is None:
            continue
        byl[x["lam"]].append(yellow_first(x))
    ls = sorted(byl)
    mu = [np.mean(byl[l]) for l in ls]
    se = []
    for l in ls:
        n = len(byl[l])
        pj = (np.sum(byl[l]) + 0.5) / (n + 1)
        se.append(np.sqrt(pj * (1 - pj) / n))
    ax.errorbar(ls, mu, yerr=se, fmt=STYLE[m], color=C[m], lw=2.0, ms=4.5, capsize=2.5, label=LBL[m])
ax.axhline(0.5, color="k", lw=.6, alpha=.4); ax.axvline(0, color="k", lw=.6, alpha=.4)
ax.set_xlabel(r"$\lambda = \log_2(r_{yellow}/r_{syco})$")
ax.set_ylabel("P(YELLOW mentioned first | both mentioned)")
ax.set_ylim(-0.03, 1.03); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=10)
ax.set_title("Pure ordering: which concept comes first when both are verbalized")
fig.tight_layout()
fig.savefig(f"{RES}/figR4_order_only.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("figR4 ok")

# ---- figR2: per-trait DIRECT presence rate vs lam (pooled over R) ----
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
for ax, trait, fld in zip(axes, ("yellow", "sycophancy"), ("fy", "fs")):
    for m, rows in data.items():
        byl = defaultdict(list)
        for x in rows:
            byl[x["lam"]].append(x[fld] is not None)
        ls = sorted(byl)
        ax.plot(ls, [np.mean(byl[l]) for l in ls], STYLE[m], color=C[m], lw=2.0, ms=4.5, label=LBL[m])
    ax.set_title(f"{trait} directly mentioned anywhere")
    ax.set_xlabel(r"$\lambda = \log_2(r_{yellow}/r_{syco})$")
    ax.grid(alpha=.3)
axes[0].set_ylabel("fraction of generations")
axes[0].set_ylim(-0.03, 1.03)
axes[0].legend(loc="lower left", fontsize=10)
fig.suptitle("Dose response: presence of each steered concept (pooled over R)")
fig.tight_layout()
fig.savefig(f"{RES}/figR2_presence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("figR2 ok")

# ---- figR3: normalized first-mention position vs lam, pooled over R ----
# A fixed absent=MISS is unfair cross-model (kitft emits ~3 lines, v3 ~10): position is
# normalized per row by the number of judged lines L, absent = 1.0, so 0 = first line.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
for ax, trait, fld in zip(axes, ("yellow", "sycophancy"), ("fy", "fs")):
    for m, rows in data.items():
        byl = defaultdict(list)
        for x in rows:
            L = len(x["yellow"])
            byl[x["lam"]].append((x[fld] - 1) / L if x[fld] is not None else 1.0)
        ls = sorted(byl)
        ax.plot(ls, [np.mean(byl[l]) for l in ls], STYLE[m], color=C[m], lw=2.0, ms=4.5, label=LBL[m])
    ax.set_title(f"first line directly mentioning {trait}")
    ax.set_xlabel(r"$\lambda = \log_2(r_{yellow}/r_{syco})$")
    ax.grid(alpha=.3)
axes[0].set_ylabel("mean normalized first-mention position\n(0 = first line, absent = 1)")
axes[0].set_ylim(1.03, -0.03)
axes[0].legend(loc="lower left", fontsize=10)
fig.suptitle("Where each steered concept first appears (pooled over R)")
fig.tight_layout()
fig.savefig(f"{RES}/figR3_first_index.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("figR3 ok")
print("RATIO_PLOT_DONE")
