"""Marginal FVE per token, forwards and backwards — ws vs suffix-RL iter_50.

Reads suffix_rl_results/sweep_dense.csv (arm,side,L,fve) and plots
marginal(L) = FVE(L) − FVE(L−1) (with FVE(0) = 0-information baseline = 0 by
definition of FVE against the mean… we anchor at FVE(1) instead: the marginal
at L=1 is FVE(1) itself). Raw curves, no smoothing (house rule from the
v1/kitft marginal plots). Two panels, one per direction — same y-scale, so
"where does each model put its information" reads directly across.

Colors: validated categorical slots — ws #2a78d6 (blue), it50 #eb6834 (orange).
Direction lives in the panel, not the hue, so each panel has exactly two series.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
D = HERE / "suffix_rl_results"
COLORS = {"ws": "#2a78d6", "it50": "#eb6834"}
LABELS = {"ws": "warm-start", "it50": "suffix-RL iter_50"}

curves: dict = {}
for row in csv.DictReader(open(D / "sweep_dense.csv")):
    curves.setdefault((row["arm"], row["side"]), {})[int(row["L"])] = float(row["fve"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, side, title, xlab in (
    (axes[0], "prefix", "forwards — marginal FVE of the L-th token from the START",
     "token position L (from start)"),
    (axes[1], "suffix", "backwards — marginal FVE of the L-th token from the END",
     "token position L (from end)"),
):
    for arm in ("ws", "it50"):
        c = curves[(arm, side)]
        Ls = sorted(c)
        marg = [c[Ls[0]]] + [c[L] - c[L - 1] for L in Ls[1:]]
        ax.plot(Ls, marg, color=COLORS[arm], lw=1.4, label=LABELS[arm])
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel(xlab, fontsize=9)
    ax.grid(True, color="#eeeeea", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("ΔFVE per token (raw, N=60, T=1)", fontsize=9)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Where the information sits: marginal round-trip FVE per token, read from each end", fontsize=11)
fig.tight_layout()
out = D / "fig_marginal_fve_bidir.png"
fig.savefig(out, dpi=180)
print(f"wrote {out}")
