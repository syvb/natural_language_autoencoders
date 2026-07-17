"""Marginal FVE per token (matryoshka 27B, clean held-out) on a SYMLOG y-axis:
linear inside |y|<1e-3, logarithmic outside — the closest thing to a log axis
that admits the (slightly) negative tail. Marginal = Δ(cumulative FVE)/Δtokens,
plotted at interval midpoints (the CSV is step-1 to 50 tokens, coarser after).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "clean_token_fve.csv")))
L = [int(r["length_tokens"]) for r in rows]
F = [float(r["fve"]) for r in rows]
xs, ys = [], []
for i in range(1, len(L)):
    dl = L[i] - L[i - 1]
    xs.append((L[i] + L[i - 1]) / 2)
    ys.append((F[i] - F[i - 1]) / dl)

PURPLE = "#9467bd"
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.plot(xs, ys, "-", color=PURPLE, lw=2.2, zorder=3)
ax.axhline(0, color="#666", lw=0.9, zorder=2)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_yticks([1e-1, 1e-2, 1e-3, 0, -1e-3])
ax.set_yticklabels(["0.1", "0.01", "0.001", "0", "−0.001"])
ax.set_xlabel("token position", fontsize=11.5)
ax.set_ylabel("marginal FVE per additional token", fontsize=11.5)
ax.set_title("Marginal FVE per token — clean held-out (matryoshka 27B)\n"
             "symlog y: linear inside ±0.001, logarithmic outside", fontsize=12)
ax.set_ylim(-2.2e-3, 0.2)
ax.grid(color="#aaa", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.02, "n = 100 held-out docs × 1 explanation; marginal = Δ(mean cumulative FVE)/Δtokens, "
                      "interval midpoints (step 1 to 50 tok, coarser beyond).",
         fontsize=8, color="#888", ha="left")
fig.tight_layout()
out = HERE / "clean_mat_marginal_fve_token_SYMLOG.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("saved", out)
