"""Blog figure: Fable 5 usefulness preference as diverging 100% stacked bars.

    python plot_usefulness_blog.py   # -> results/fig_usefulness_blog.png
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#22262b", "#6b7280", "#e5e7eb"
MAT, STD = "#2a78d6", "#e34948"
GROUPS = [("all", "All contexts"),
          ("pretrain", "Pretraining documents"),
          ("wildchat", "WildChat conversations")]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    man = {c["ci"]: c["domain"] for c in
           json.load(open(HERE / "data" / "manifest_quality.json"))["contexts"]}
    rows = json.load(open(HERE / "results" / "fable5_native_paired.json"))["paired"]

    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=200)
    ys = [2, 1, 0]
    h = 0.52
    for y, (g, label) in zip(ys, GROUPS):
        cnt = Counter(r["winner"] for r in rows if g == "all" or man[r["ci"]] == g)
        n = cnt["mat"] + cnt["std"]
        p, lo, hi = wilson(cnt["mat"], n)
        m = p * 100
        ax.barh(y, m, h, color=MAT, zorder=3)
        ax.barh(y, 100 - m, h, left=m, color=STD, zorder=3)
        ax.errorbar(m, y, xerr=[[m - lo * 100], [hi * 100 - m]],
                    color="white", lw=1.6, capsize=3.5, zorder=5, alpha=0.95)
        ax.text(m / 2, y, f"{m:.0f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(m + (100 - m) / 2, y, f"{100 - m:.0f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(-2, y, label, ha="right", va="center", fontsize=11.5)

    ax.axvline(50, color=INK, ls=(0, (2, 3)), lw=1.3, zorder=4, alpha=0.85)
    ax.text(50, 2.62, "even split", ha="center", fontsize=9.5, color=MUTED,
            style="italic")

    # arm labels above the top bar, colored — the legend
    ax.text(0, 3.15, "Matryoshka wins", ha="left", fontsize=12,
            fontweight="bold", color=MAT)
    ax.text(100, 3.15, "Standard wins", ha="right", fontsize=12,
            fontweight="bold", color=STD)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 3.45)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    ax.set_title("Explanation usefulness", fontsize=18, fontweight="bold",
                 loc="left", pad=34, x=-0.31)
    fig.text(0.001, -0.04,
             "Claude Fable 5 picks the more useful of two explanations of the same "
             "Qwen3.6-27B activation, shown the source passage\nand its true "
             "continuation. “Useful” = helps you understand the model’s "
             "internals, not readability. Forced choice, both presentation\n"
             "orders, 100 contexts (50 per domain). White whisker: 95% CI on the "
             "split.",
             fontsize=9, color=MUTED, ha="left", va="top", linespacing=1.55)
    fig.tight_layout()
    out = HERE / "results" / "fig_usefulness_blog.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
