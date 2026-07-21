"""Head-to-head: matryoshka in its TRAINED order vs standard with its lines
REVERSED — final-token-revealed suffix prediction, nex-n2-mini judge.

The fairest anytime comparison available to the standard model: its best
(reversed) reading order against the matryoshka's native one.

    python plot_ft_mat_vs_stdrev.py   # -> ft_mat_vs_stdrev.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"


def curve(report, tag):
    c = report["arms"][tag]["curve"]
    ks = sorted(int(k) for k in c)
    return (ks, [c[str(k)]["acc"] * 100 for k in ks],
            [c[str(k)]["ci"][0] * 100 for k in ks],
            [c[str(k)]["ci"][1] * 100 for k in ks])


def main():
    ft = json.load(open(HERE / "results" / "ft_results_nex.json"))
    rev = json.load(open(HERE / "results" / "ft_results_rev_nex.json"))
    tko = ft["token_only"]["acc"] * 100

    series = [("mat (trained order)", curve(ft, "mat"), BLUE),
              ("std (lines reversed)", curve(rev, "std"), RED)]
    solid = {name: dict(zip(ks, ys)) for name, (ks, ys, _, _), _ in series}

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    for name, (ks, ys, lo, hi), col in series:
        ax.fill_between(ks, lo, hi, color=col, alpha=0.12, zorder=2)
        ax.plot(ks, ys, "o-", color=col, lw=2.7, ms=7, zorder=4)
        other = solid["std (lines reversed)" if name.startswith("mat") else "mat (trained order)"]
        for k, y in zip(ks, ys):
            above = y >= other.get(k, y - 1)
            ax.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                        xytext=(0, 9 if above else -16), ha="center",
                        fontsize=9, color=col, fontweight="bold", zorder=5)

    ax.axhline(tko, color=GRAY, ls="-", lw=2.2, alpha=0.75, zorder=3)
    ax.text(256, tko + 1.5, f"final token ALONE: {tko:.0f}%", ha="right",
            fontsize=10, color=GRAY, fontweight="bold")
    ax.axhline(10, color=INK, ls=":", lw=1.3, zorder=1)
    ax.text(4, 11.2, "chance 10%", ha="left", fontsize=9, color=INK)

    ks = series[0][1][0]
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(ks[0] * 0.88, ks[-1] * 1.1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("explanation content tokens the judge may read (first N)", fontsize=11.5)
    ax.set_ylabel("blind judge accuracy (%)", fontsize=11.5)
    ax.set_title("Matryoshka (trained order) vs standard READ BACK-TO-FRONT\n"
                 "final token revealed, same-document distractors — Qwen3.6-27B, "
                 "nex-n2-mini judge", fontsize=12.5, fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o",
               label="Matryoshka, trained order + final token"),
        Line2D([], [], color=RED, lw=2.6, marker="o",
               label="Standard, lines reversed + final token"),
        Line2D([], [], color=GRAY, lw=2.2, alpha=0.75, label="final token only"),
    ], fontsize=9.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRAY, alpha=0.22, zorder=0)
    fig.text(0.01, -0.02,
             "250 held-out Ultra-FineWeb contexts, 10-way choice among same-document "
             "windows.\nTruncation takes the FIRST N tokens of each arm's text: the "
             "matryoshka's native opening vs the standard explanation's original tail.",
             fontsize=8.2, color=GRAY, ha="left", va="top")
    fig.tight_layout()
    out = HERE / "ft_mat_vs_stdrev.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
