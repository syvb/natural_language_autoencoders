"""Cumulative suffix FVE: reconstruction from the last k tokens, k=1..120 —
the integral view of fig_marginal_fve_bidir's backwards panel, and (linearly
k-weighted) the quantity suffix-RL actually optimizes. Reads sweep_dense.csv.
Zero-crossings annotated: the k at which reading from the end stops being
worse than predicting the mean."""
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
    if row["side"] == "suffix":
        curves.setdefault(row["arm"], {})[int(row["L"])] = float(row["fve"])

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for arm in ("ws", "it50"):
    c = curves[arm]
    Ls = sorted(c)
    ys = [c[L] for L in Ls]
    ax.plot(Ls, ys, color=COLORS[arm], lw=2, label=LABELS[arm])
    cross = next((L for L in Ls[1:] if c[L] >= 0 and c[L - 1] < 0), None)
    if cross:
        ax.annotate(f"crosses 0 at k={cross}", (cross, 0), xytext=(cross + 4, -0.09 if arm == "it50" else -0.17),
                    fontsize=8.5, color=COLORS[arm],
                    arrowprops=dict(arrowstyle="-", color=COLORS[arm], lw=0.8))
ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
ax.set_xlabel("k — tokens read, starting from the FINAL token", fontsize=9.5)
ax.set_ylabel("round-trip FVE (last k tokens → critic)", fontsize=9.5)
ax.set_title("Cumulative FVE reading backwards from the end (N=60, T=1, cap 120)", fontsize=10.5)
ax.grid(True, color="#eeeeea", lw=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
out = D / "fig_cumulative_suffix_fve.png"
fig.savefig(out, dpi=180)
print(f"wrote {out}")
