"""Standalone same-document budget curve — the specificity crossover on its own.

Single panel, same-document distractors only (no off-document reference), with
the budget axis extended past 120 tokens (std explanations median ~163 tokens,
so T=120 truncates all of them; T=256 is effectively the full explanation).

    python plot_specificity_budget.py   # results/{budget_hard,results_hard}.json -> specificity_budget.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"


def main():
    b = json.load(open(HERE / "results" / "budget_hard.json"))
    h = json.load(open(HERE / "results" / "results_hard.json"))
    chance = 100.0 / h["meta"]["n_options"]

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    curves = {}
    for tag in ("mat", "std"):
        c = b["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        curves[tag] = (ks,
                       [c[str(k)]["acc"] * 100 for k in ks],
                       [c[str(k)]["ci"][0] * 100 for k in ks],
                       [c[str(k)]["ci"][1] * 100 for k in ks])

    for tag, col, lab in [("mat", BLUE, "Matryoshka NLA"), ("std", RED, "Standard NLA")]:
        ks, ys, lo, hi = curves[tag]
        other = curves["std" if tag == "mat" else "mat"][1]
        ax.fill_between(ks, lo, hi, color=col, alpha=0.13, zorder=2)
        ax.plot(ks, ys, "o-", color=col, lw=2.7, ms=7.5, label=lab, zorder=4)
        for i, (k, y) in enumerate(zip(ks, ys)):
            above = y >= other[i]
            ax.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                        xytext=(0, 9 if above else -17), ha="center",
                        fontsize=9.5, color=col, fontweight="bold", zorder=5)

    # reference lines
    ax.axhline(chance, color=INK, ls=":", lw=1.4, zorder=1)
    ax.text(16, chance + 1.6, f"chance {chance:.0f}%", ha="center", fontsize=9.5, color=INK)
    sk = h["skyline"]["acc"] * 100
    ax.axhline(sk, color=GRAY, ls="-.", lw=1.4, alpha=0.85, zorder=1)
    ax.text(curves["mat"][0][0], sk + 1.6, f"skyline (grader sees the passage): {sk:.0f}%",
            ha="left", fontsize=9.5, color=GRAY)

    # mark the training-truncation ceiling
    ax.axvline(120, color=GRAY, ls=":", lw=1.2, alpha=0.8, zorder=1)
    ax.text(120, 3, " top of matryoshka's U[1,120]\n truncation training range",
            ha="left", va="bottom", fontsize=8.5, color=GRAY)

    ks = curves["mat"][0]
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(ks[0] * 0.88, ks[-1] * 1.1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("explanation content tokens the grader may read (first N)", fontsize=12)
    ax.set_ylabel("blind grader accuracy (%)", fontsize=12)
    ax.set_title("Specificity under same-document distractors\n"
                 "identify the exact continuation among 10 windows of the same document — Qwen3.6-27B, layer 42",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=11.5, loc="center right", bbox_to_anchor=(0.98, 0.42))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRAY, alpha=0.22, zorder=0)
    fig.text(0.01, -0.02,
             "250 held-out Ultra-FineWeb contexts, Haiku-4.5 grader, 10-way choice, all options from the true answer's own "
             "document. Shaded = 95% Wilson CI (n=500/point). T=256 ≈ full explanation (median length: matryoshka 171 tok, standard 163).",
             fontsize=8.2, color=GRAY, ha="left")
    fig.tight_layout()
    out = HERE / "specificity_budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
