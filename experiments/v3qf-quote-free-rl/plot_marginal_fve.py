"""Mean marginal round-trip FVE per line: control pair vs quote-penalty pair.

Reads the two fve_by_line jsons (50 held-out docs each, same seed/doc selection,
each pair scored by its own co-trained critic). Marginal of line k = FVE@k −
FVE@(k−1), with FVE@0 = 0 (predicting the population mean). Renders two
figures: linear y and log y (log panel drops non-positive means — noted on the
figure). Run locally; matplotlib only.

Usage: python plot_marginal_fve.py fve_by_line_v3_t1_50.json fve_by_line_v3qf_50.json
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONTROL, QF = sys.argv[1], sys.argv[2]
KMAX = 12          # line index cap (deeper lines exist in <~30% of samples)
MIN_N = 15         # need at least this many samples contributing at index k

SERIES = [
    (CONTROL, "control pair (no penalty)", "#666666", "--", "o"),
    (QF, "quote-penalty pair", "#9467bd", "-", "s"),
]


def marginals(path):
    """rows: per-example marginal ΔFVE list; m_1 = FVE@1 (FVE@0 = 0 by defn)."""
    exs = json.load(open(path))["examples"]
    rows = []
    for e in exs:
        f = e["fve_by_k"]
        rows.append([f[0]] + [f[k] - f[k - 1] for k in range(1, len(f))])
    return rows


def mean_sem(rows):
    ks, mean, sem, ns = [], [], [], []
    for k in range(KMAX):
        vals = np.array([r[k] for r in rows if len(r) > k])
        if len(vals) < MIN_N:
            break
        ks.append(k + 1)
        mean.append(vals.mean())
        sem.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        ns.append(len(vals))
    return np.array(ks), np.array(mean), np.array(sem), ns


curves = [(label, c, ls, mk, *mean_sem(marginals(p))) for p, label, c, ls, mk in SERIES]

for logy, fname in [(False, "marginal_fve_per_line.png"),
                    (True, "marginal_fve_per_line_LOGY.png")]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, c, ls, mk, ks, mean, sem, ns in curves:
        if logy:
            keep = mean > 0
            ax.plot(ks[keep], mean[keep], ls, color=c, marker=mk, ms=5, lw=2, label=label)
            ax.fill_between(ks[keep], np.clip(mean[keep] - sem[keep], 1e-4, None),
                            mean[keep] + sem[keep], color=c, alpha=0.15, lw=0)
        else:
            ax.plot(ks, mean, ls, color=c, marker=mk, ms=5, lw=2, label=label)
            ax.fill_between(ks, mean - sem, mean + sem, color=c, alpha=0.15, lw=0)
    if logy:
        ax.set_yscale("log")
        ax.set_title("Marginal round-trip FVE per line (log scale)", fontsize=12)
        ax.text(0.98, 0.02, "non-positive means omitted", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#888888")
    else:
        ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_title("Marginal round-trip FVE per line", fontsize=12)
    ax.set_xlabel("line index of the verbalization")
    ax.set_ylabel("mean ΔFVE from adding this line")
    ax.set_xticks(range(1, KMAX + 1))
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    n0 = curves[0][7][0] if curves[0][7] else 0
    fig.text(0.01, 0.01,
             f"{n0} held-out activations per series, same docs/seed; each pair scored by its own "
             "co-trained critic; shaded = ±SEM", fontsize=7.5, color="#888888")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(fname, dpi=180)
    print("wrote", fname)
