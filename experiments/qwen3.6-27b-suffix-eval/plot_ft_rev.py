"""Final-token-revealed suffix prediction with explanation LINES REVERSED.

Backloading probe on the judge-side eval: solid = lines-reversed explanation +
final token; faded dashed = original order + final token (same judge's prior
run); gray line = final token alone.

    python plot_ft_rev.py                    # nex judge -> ft_reversed.png
    python plot_ft_rev.py --grader haiku     # Haiku variant
"""
import argparse
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grader", choices=["nex", "haiku"], default="nex")
    args = ap.parse_args()
    if args.grader == "nex":
        rev_f, ft_f, jname, out_name = ("ft_results_rev_nex.json", "ft_results_nex.json",
                                        "nex-n2-mini", "ft_reversed.png")
    else:
        rev_f, ft_f, jname, out_name = ("ft_results_rev.json", "ft_results.json",
                                        "Haiku-4.5", "ft_reversed_haiku.png")
    rev = json.load(open(HERE / "results" / rev_f))
    ft = json.load(open(HERE / "results" / ft_f))
    tko = rev["token_only"]["acc"] * 100

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    solid = {}
    for tag in ("mat", "std"):
        c = rev["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        solid[tag] = dict(zip(ks, [c[str(k)]["acc"] * 100 for k in ks]))
    for tag, col in [("mat", BLUE), ("std", RED)]:
        # original order (faded dashed)
        c = ft["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        ax.plot(ks, [c[str(k)]["acc"] * 100 for k in ks], "--", color=col,
                lw=1.6, alpha=0.35, zorder=2)
        # reversed (solid)
        c = rev["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        ys = [c[str(k)]["acc"] * 100 for k in ks]
        lo = [c[str(k)]["ci"][0] * 100 for k in ks]
        hi = [c[str(k)]["ci"][1] * 100 for k in ks]
        ax.fill_between(ks, lo, hi, color=col, alpha=0.12, zorder=2)
        ax.plot(ks, ys, "o-", color=col, lw=2.7, ms=7, zorder=4)
        other = solid["std" if tag == "mat" else "mat"]
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

    ks = sorted(solid["mat"])
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(ks[0] * 0.88, ks[-1] * 1.1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("explanation content tokens the judge may read (first N)", fontsize=11.5)
    ax.set_ylabel("blind judge accuracy (%)", fontsize=11.5)
    ax.set_title("Suffix prediction, final token revealed, explanation LINES REVERSED\n"
                 f"same-document distractors — Qwen3.6-27B, layer 42, {jname} judge",
                 fontsize=12.5, fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o", label="Matryoshka, lines reversed + final token"),
        Line2D([], [], color=RED, lw=2.6, marker="o", label="Standard, lines reversed + final token"),
        Line2D([], [], color=INK, lw=1.6, ls="--", alpha=0.4, label="original order + final token"),
        Line2D([], [], color=GRAY, lw=2.2, alpha=0.75, label="final token only"),
    ], fontsize=9.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRAY, alpha=0.22, zorder=0)
    fig.text(0.01, -0.02,
             f"250 held-out Ultra-FineWeb contexts, {jname} judge, 10-way choice among "
             "same-document windows.\nTruncation takes the FIRST N tokens of the reversed "
             "text, so low budgets read each explanation's ORIGINAL tail.",
             fontsize=8.2, color=GRAY, ha="left", va="top")
    fig.tight_layout()
    out = HERE / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")

    for tag in ("mat", "std"):
        a = rev["arms"][tag]
        b = ft["arms"][tag]
        print(f"{tag}: rev full {a['full']['acc']:.1%} (orig {b['full']['acc']:.1%}) | "
              f"rev T4 {a['curve']['4']['acc']:.1%} (orig {b['curve']['4']['acc']:.1%})")


if __name__ == "__main__":
    main()
