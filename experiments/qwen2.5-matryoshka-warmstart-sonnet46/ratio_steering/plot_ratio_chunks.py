"""Word-chunk control analysis: kitft judged on 10 equal-word chunks vs v3 on its lines.

Removes the granularity confound (kitft ~3 long lines vs v3 ~10 short ones): with 10
chunks both models have the same order resolution, and the 200-char judge clip never
bites (chunks are ~65 chars). Prints pooled / order-only / ties-excluded slopes and
renders figR5 (pooled psychometric) + figR6 (order-only), with kitft-on-lines as a
light reference curve.
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
SERIES = [
    ("v3", "ratio_judged_v3.json", "matryoshka NLA (ours), lines", "#9467bd", "D-", 2.3, 1.0),
    ("kitft_chunks", "ratio_judged_kitft_chunks.json", "kitft baseline, 10 word-chunks", "#444444", "s--", 2.0, 1.0),
    ("kitft", "ratio_judged_kitft.json", "kitft baseline, lines (reference)", "#bbbbbb", "s:", 1.4, 0.8),
]


def first_idx(labels):
    for i, l in enumerate(labels):
        if l == 2:
            return i + 1
    return None


def yellow_first(x):
    fy, fs = x["fy"], x["fs"]
    if fy is None and fs is None:
        return None
    if fs is None:
        return 1.0
    if fy is None:
        return 0.0
    return 1.0 if fy < fs else (0.0 if fs < fy else 0.5)


def logistic_slope(lams, ys):
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


data = {}
for key, fn, *_ in SERIES:
    rows = json.load(open(f"{RES}/{fn}"))
    for x in rows:
        x["fy"], x["fs"] = first_idx(x["yellow"]), first_idx(x["syco"])
    data[key] = rows

print("slopes (pooled over R):")
for key, rows in data.items():
    pooled = [(x["lam"], v) for x in rows if (v := yellow_first(x)) is not None]
    a, b = logistic_slope([p[0] for p in pooled], [p[1] for p in pooled])
    both = [(x["lam"], yellow_first(x)) for x in rows if x["fy"] is not None and x["fs"] is not None]
    _, bb = logistic_slope([p[0] for p in both], [p[1] for p in both])
    nt = [(l, v) for l, v in pooled if v != 0.5]
    _, bn = logistic_slope([p[0] for p in nt], [p[1] for p in nt])
    ties = sum(1 for _, v in pooled if v == 0.5)
    print(f"  {key:13s} pooled={b:+.3f} (mid {-a/b:+.2f}, n={len(pooled)}) | "
          f"order-only={bb:+.3f} (n={len(both)}) | no-ties={bn:+.3f} | tie rate {ties/len(pooled):.3f}")

for figname, title, cond in (
        ("figR5_chunks_pooled.png", "Which steered concept is verbalized first (pooled over R)", "pooled"),
        ("figR6_chunks_order_only.png", "Pure ordering: which concept comes first when both are verbalized", "both")):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for key, fn, lbl, c, style, lw, alpha in SERIES:
        byl = defaultdict(list)
        for x in data[key]:
            if cond == "both" and (x["fy"] is None or x["fs"] is None):
                continue
            v = yellow_first(x)
            if v is not None:
                byl[x["lam"]].append(v)
        ls = sorted(l for l in byl if len(byl[l]) >= 8)
        mu = [np.mean(byl[l]) for l in ls]
        se = [np.sqrt(max(np.mean(byl[l]) * (1 - np.mean(byl[l])), 1e-9) / len(byl[l])) for l in ls]
        ax.errorbar(ls, mu, yerr=se, fmt=style, color=c, lw=lw, ms=4.5, capsize=2.5, alpha=alpha, label=lbl)
    ax.axhline(0.5, color="k", lw=.6, alpha=.4); ax.axvline(0, color="k", lw=.6, alpha=.4)
    ax.set_xlabel(r"$\lambda = \log_2(r_{yellow}/r_{syco})$")
    ax.set_ylabel("P(YELLOW first)" if cond == "pooled" else "P(YELLOW first | both mentioned)")
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(f"{RES}/{figname}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{figname} ok")
print("RATIO_CHUNK_PLOT_DONE")
