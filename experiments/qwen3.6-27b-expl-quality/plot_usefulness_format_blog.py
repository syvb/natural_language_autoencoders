"""Blog figure: Fable 5 usefulness preference, native vs format-shifted
(matryoshka reformatted into prose; the standard is prose natively).

    python plot_usefulness_format_blog.py  # -> results/fig_usefulness_format_blog.png
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK, MUTED = "#22262b", "#6b7280"
MAT, STD = "#2a78d6", "#e34948"
ROWS = [("fable5_native_paired.json", "Native formats\n(list vs. prose)"),
        ("fable5_prose_paired.json", "Matryoshka rewritten\nas prose")]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    fig, ax = plt.subplots(figsize=(8.6, 2.7), dpi=200)
    ys = [1, 0]
    h = 0.5
    for y, (fn, label) in zip(ys, ROWS):
        rows = json.load(open(HERE / "results" / fn))["paired"]
        cnt = Counter(r["winner"] for r in rows)
        n = cnt["mat"] + cnt["std"]
        p, lo, hi = wilson(cnt["mat"], n)
        m = p * 100
        ax.barh(y, m, h, color=MAT, zorder=3)
        ax.barh(y, 100 - m, h, left=m, color=STD, zorder=3)
        ax.errorbar(m, y, xerr=[[m - lo * 100], [hi * 100 - m]],
                    color="white", lw=1.6, capsize=3.5, zorder=5, alpha=0.95)
        ax.text(m / 2, y, f"{m:.0f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(m + (100 - m) / 2, y, f"{100 - m:.0f}%", ha="center",
                va="center", fontsize=13, fontweight="bold", color="white",
                zorder=4)
        ax.text(-2, y, label, ha="right", va="center", fontsize=11.5,
                linespacing=1.3)

    ax.axvline(50, color=INK, ls=(0, (2, 3)), lw=1.3, zorder=4, alpha=0.85)
    ax.text(50, 1.58, "even split", ha="center", fontsize=9.5, color=MUTED,
            style="italic")
    ax.text(0, 2.05, "Matryoshka wins", ha="left", fontsize=12,
            fontweight="bold", color=MAT)
    ax.text(100, 2.05, "Standard wins", ha="right", fontsize=12,
            fontweight="bold", color=STD)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.35)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Explanation usefulness", fontsize=18, fontweight="bold",
                 loc="left", pad=34, x=-0.28)
    fig.tight_layout()
    out = HERE / "results" / "fig_usefulness_format_blog.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
