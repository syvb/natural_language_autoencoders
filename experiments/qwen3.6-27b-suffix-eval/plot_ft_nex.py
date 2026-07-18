"""Standalone nex-n2-mini panel of the final-token-revealed eval.

    python plot_ft_nex.py   # -> ft_nex.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_ft_graders import panel, L, HERE, BLUE, RED, INK, GRAY


def main():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    panel(ax, L("ft_results_nex.json"), L("budget_hard_nex.json"),
          "Final token revealed to the judge — nex-n2-mini\n"
          "same-document suffix prediction, Qwen3.6-27B L42")
    ax.set_ylabel("blind judge accuracy (%)", fontsize=11.5)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o", label="Matryoshka + final token"),
        Line2D([], [], color=RED, lw=2.6, marker="o", label="Standard + final token"),
        Line2D([], [], color=INK, lw=1.6, ls="--", alpha=0.4, label="explanation only"),
        Line2D([], [], color=GRAY, lw=2.0, alpha=0.75, label="final token only"),
    ], fontsize=10, loc="lower right")
    fig.tight_layout()
    out = HERE / "ft_nex.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
