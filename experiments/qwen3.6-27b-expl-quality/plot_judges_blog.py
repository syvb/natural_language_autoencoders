"""Blog-quality version of the judge-comparison figure.

    python plot_judges_blog.py   # -> results/fig_judge_compare_blog.png
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
GEMMA, FABLE = "#0f8a80", "#7a52c7"
GROUPS = [("all", "All contexts", 500),
          ("pretrain", "Pretraining\ndocuments", 250),
          ("wildchat", "WildChat\nconversations", 250)]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
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
    runs = [
        ("Gemma 4 31B judge", GEMMA,
         json.load(open(HERE / "results" / "transplant_mixed_paired.json")),
         "winner"),
        ("Claude Fable 5 judge", FABLE,
         json.load(open(HERE / "results" / "fable5_native_paired.json"))["paired"],
         "winner"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
    w = 0.3
    for j, (jname, col, rows, key) in enumerate(runs):
        for i, (g, label, _) in enumerate(GROUPS):
            cnt = Counter(r[key] for r in rows if g == "all" or man[r["ci"]] == g)
            n = cnt["mat"] + cnt["std"]
            p, lo, hi = wilson(cnt["mat"], n)
            x = i + (j - 0.5) * (w + 0.04)
            ax.bar(x, p * 100, w, color=col, zorder=3,
                   label=jname if i == 0 else None)
            ax.errorbar(x, p * 100, [[(p - lo) * 100], [(hi - p) * 100]],
                        color=INK, lw=1.1, capsize=2.5, alpha=0.55, zorder=4)
            ax.annotate(f"{p*100:.0f}%", (x, hi * 100 + 2),
                        ha="center", fontsize=12, fontweight="bold", color=col)

    ax.axhline(50, color=INK, ls=(0, (2, 3)), lw=1.2, zorder=2, alpha=0.8)

    ax.set_xticks(range(3), [l for _, l, _ in GROUPS], fontsize=11.5)
    ax.set_yticks(range(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)],
                  fontsize=10)
    ax.set_ylim(0, 82)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("head-to-head wins for the matryoshka explanation",
                  fontsize=11.5, labelpad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.legend(fontsize=11, loc="upper left", frameon=False,
              handlelength=1.2, handleheight=1.1, borderaxespad=0)

    ax.set_title(
        "Which explanation style is more useful?\nDepends on whom you ask.",
        fontsize=17, fontweight="bold", loc="left", pad=18)
    fig.text(0.001, -0.03,
             "Blind LLM judges pick the more useful of two explanations of the same "
             "Qwen3.6-27B activation (matryoshka vs. standard),\nshown the source "
             "passage and its true continuation. Forced choice, both presentation "
             "orders. Error bars: 95% CI over judgments.\nDotted line: 50%, where "
             "a judge is indifferent between the two styles.",
             fontsize=9, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.tight_layout()
    out = HERE / "results" / "fig_judge_compare_blog.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
