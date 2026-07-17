"""Final-token-revealed suffix prediction (same-doc): does the matryoshka's
early advantage survive when the judge is handed the passage's final token?

Solid = judge sees explanation + final token; faded dashed = explanation only
(budget_hard.json); gray line = final token alone (no explanation).

    python plot_ft.py    # -> ft_check.png
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


def main():
    ft = json.load(open(HERE / "results" / "ft_results.json"))
    nt = json.load(open(HERE / "results" / "budget_hard.json"))
    tko = ft["token_only"]["acc"] * 100

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    curves = {}
    for tag, col in [("mat", BLUE), ("std", RED)]:
        # without token (faded dashed)
        c = nt["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        ax.plot(ks, [c[str(k)]["acc"] * 100 for k in ks], "--", color=col,
                lw=1.6, alpha=0.35, zorder=2)
        # with token (solid)
        c = ft["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        ys = [c[str(k)]["acc"] * 100 for k in ks]
        lo = [c[str(k)]["ci"][0] * 100 for k in ks]
        hi = [c[str(k)]["ci"][1] * 100 for k in ks]
        curves[tag] = dict(zip(ks, ys))
        ax.fill_between(ks, lo, hi, color=col, alpha=0.12, zorder=2)
        ax.plot(ks, ys, "o-", color=col, lw=2.7, ms=7, zorder=4)
        other_first = 27.6 if tag == "mat" else 51.6
        for k, y in zip(ks, ys):
            above = (tag == "mat") == (y >= curves.get("std", {}).get(k, other_first) if tag == "mat" else True)
            dy = 9 if tag == "mat" else -16
            ax.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=9,
                        color=col, fontweight="bold", zorder=5)

    ax.axhline(tko, color=GRAY, ls="-", lw=2.2, alpha=0.75, zorder=3)
    ax.text(256, tko + 1.5, f"final token ALONE: {tko:.0f}%", ha="right",
            fontsize=10, color=GRAY, fontweight="bold")
    ax.axhline(10, color=INK, ls=":", lw=1.3, zorder=1)
    ax.text(4, 11.2, "chance 10%", ha="left", fontsize=9, color=INK)

    # callout: gap decomposition at T=4 (no arrow — the text names the point)
    ax.text(4.0, 88, "at 4 tokens the matryoshka–standard gap is still +24 pts\n"
                     "(was +36 without the token: ~1/3 of the edge was token-restating;\n"
                     "the matryoshka curve itself barely moves — the token was redundant for it)",
            fontsize=9.6, color=INK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GRAY, alpha=0.92))

    ks = sorted(curves["mat"])
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(ks[0] * 0.88, ks[-1] * 1.1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("explanation content tokens the judge may read (first N)", fontsize=11.5)
    ax.set_ylabel("blind judge accuracy (%)", fontsize=11.5)
    ax.set_title("Suffix prediction with the final token REVEALED to the judge\n"
                 "same-document distractors — Qwen3.6-27B, layer 42",
                 fontsize=12.5, fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o", label="Matryoshka + final token"),
        Line2D([], [], color=RED, lw=2.6, marker="o", label="Standard + final token"),
        Line2D([], [], color=INK, lw=1.6, ls="--", alpha=0.4, label="explanation only (prev. result)"),
        Line2D([], [], color=GRAY, lw=2.2, alpha=0.75, label="final token only"),
    ], fontsize=9.5, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRAY, alpha=0.22, zorder=0)
    fig.text(0.01, -0.02,
             "250 held-out Ultra-FineWeb contexts, Haiku-4.5 judge, 10-way choice among same-document windows.\n"
             "Full-length (T=256): matryoshka 66.0/67.4, standard 75.6/76.8 — the specificity reversal is unaffected by the token.",
             fontsize=8.2, color=GRAY, ha="left", va="top")
    fig.tight_layout()
    out = HERE / "ft_check.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
