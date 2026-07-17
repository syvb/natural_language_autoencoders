"""Standalone front-loading graph: suffix-prediction accuracy vs. explanation
content-token budget. The headline result, on its own. No GPU.

    python plot_frontload.py    # results/budget.json (+ results.json) -> frontload.png
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
    b = json.load(open(HERE / "results" / "budget.json"))
    r = json.load(open(HERE / "results" / "results.json"))
    chance = 100.0 / r["meta"]["n_options"]

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    series = [("mat", BLUE, "Matryoshka NLA", 9), ("std", RED, "Standard NLA", -16)]
    pts = {}
    for tag, c, lab, dy in series:
        curve = b["arms"][tag]["curve"]
        ks = sorted(int(k) for k in curve)
        ys = [curve[str(k)]["acc"] * 100 for k in ks]
        los = [curve[str(k)]["ci"][0] * 100 for k in ks]
        his = [curve[str(k)]["ci"][1] * 100 for k in ks]
        pts[tag] = dict(zip(ks, ys))
        ax.fill_between(ks, los, his, color=c, alpha=0.14, zorder=2)
        ax.plot(ks, ys, "o-", color=c, lw=2.8, ms=8, label=lab, zorder=4)
        for k, y in zip(ks, ys):
            ax.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=9.5,
                        color=c, fontweight="bold", zorder=5)

    ax.axhline(chance, color=INK, ls=":", lw=1.4, zorder=1)
    ax.text(b["budgets"][0], chance + 1.8, f"chance = {chance:.0f}%",
            color=INK, fontsize=10, ha="left", va="bottom")

    # callout on the low-budget gap
    k0 = b["budgets"][0]
    ax.annotate(
        f"at {k0} tokens the matryoshka is already\nat {pts['mat'][k0]:.0f}% — the standard is at {pts['std'][k0]:.0f}%",
        xy=(k0, pts["std"][k0]), xytext=(9.5, 40),
        fontsize=10.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRAY, alpha=0.9))

    ax.set_xscale("log")
    ax.set_xticks(b["budgets"])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(b["budgets"][0] * 0.9, b["budgets"][-1] * 1.08)
    ax.set_ylim(0, 105)
    ax.set_xlabel("explanation content tokens the grader may read  (first N)", fontsize=12)
    ax.set_ylabel("blind grader accuracy  (%)", fontsize=12)
    ax.set_title("The matryoshka NLA front-loads the answer\n"
                 "Suffix prediction from a truncated activation explanation "
                 "— Qwen3.6-27B, layer 42",
                 fontsize=13.5, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRAY, alpha=0.22, zorder=0)
    # subtle footnote
    fig.text(0.01, -0.02,
             f"250 held-out Ultra-FineWeb contexts, Haiku-4.5 grader, 10-way choice. "
             f"Shaded = 95% Wilson CI (n=500/point). U[1,120] = the matryoshka's RL truncation range.",
             fontsize=8.5, color=GRAY, ha="left")
    fig.tight_layout()
    out = HERE / "frontload.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
