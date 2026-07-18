"""Marginal FVE per token (27B, clean held-out), matryoshka + standard, SYMLOG y
(linear inside |y|<1e-3, log outside — as log as the negative values allow).
Marginal = Δ(mean cumulative FVE)/Δtokens at interval midpoints.
NB the standard's early marginals are LARGE positives only because its cumulative
FVE starts deeply negative (≈−0.88 at 1 token) and climbs; the matryoshka starts
near zero and climbs to ~0.44 by token 10.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED = "#2a78d6", "#e34948"


def marginals(fname):
    rows = list(csv.DictReader(open(HERE / fname)))
    L = [int(r["length_tokens"]) for r in rows]
    F = [float(r["fve"]) for r in rows]
    xs, ys = [], []
    for i in range(1, len(L)):
        xs.append((L[i] + L[i - 1]) / 2)
        ys.append((F[i] - F[i - 1]) / (L[i] - L[i - 1]))
    return xs, ys


fig, ax = plt.subplots(figsize=(9.4, 5.6))
for fname, col, lab in [("clean_token_fve.csv", BLUE, "Matryoshka"),
                        ("clean_std_token_fve.csv", RED, "Standard")]:
    xs, ys = marginals(fname)
    ax.plot(xs, ys, "-", color=col, lw=2.1, label=lab, zorder=3)
ax.axhline(0, color="#666", lw=0.9, zorder=2)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_yticks([1e-1, 1e-2, 1e-3, 0, -1e-3, -1e-2])
ax.set_yticklabels(["0.1", "0.01", "0.001", "0", "−0.001", "−0.01"])
ax.set_ylim(-2e-2, 0.25)
ax.set_xlabel("token position", fontsize=11.5)
ax.set_ylabel("marginal FVE per additional token", fontsize=11.5)
ax.set_title("Marginal FVE per token — clean held-out (Qwen3.6-27B NLAs)\n"
             "symlog y: linear inside ±0.001, logarithmic outside", fontsize=12)
ax.legend(fontsize=11, loc="upper right")
ax.grid(color="#aaa", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.02,
         "n = 100 held-out docs × 1 explanation each; marginal = Δ(mean cumulative FVE)/Δtokens, interval midpoints "
         "(step 1 to 50 tok, coarser beyond).\nNB the standard's large early marginals climb out of a deeply negative "
         "start (cumulative FVE ≈ −0.88 at 1 token, still <0 until ~70 tokens); the matryoshka starts near 0.",
         fontsize=8, color="#888", ha="left", va="top")
fig.tight_layout()
out = HERE / "clean_marginal_fve_token_SYMLOG_both.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("saved", out)
