"""The specificity reversal (full explanation): off-document vs same-document.

When distractors are off-document the task is topic-separable and both NLAs hit
~99%. When they come from the SAME document (topic controlled), accuracy
collapses and the ranking flips — the standard NLA becomes the more specific one.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"


def main():
    o = json.load(open(HERE / "results" / "results.json"))
    h = json.load(open(HERE / "results" / "results_hard.json"))
    chance = 100.0 / o["meta"]["n_options"]

    rows = [("Skyline\n(sees passage)", o["skyline"]["acc"], h["skyline"]["acc"], GRAY),
            ("Matryoshka NLA", o["arms"]["mat"]["primary"]["acc"], h["arms"]["mat"]["primary"]["acc"], BLUE),
            ("Standard NLA", o["arms"]["std"]["primary"]["acc"], h["arms"]["std"]["primary"]["acc"], RED)]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = range(len(rows))
    w = 0.36
    for k, (lab, vo, vh, c) in zip(x, rows):
        ax.bar(k - w / 2, vo * 100, w, color=c, alpha=0.32, hatch="///", edgecolor=c, linewidth=1.2, zorder=3)
        ax.bar(k + w / 2, vh * 100, w, color=c, alpha=0.95, edgecolor=c, linewidth=1.2, zorder=3)
        ax.text(k - w / 2, vo * 100 + 1.4, f"{vo*100:.0f}", ha="center", fontsize=11, color=c, fontweight="bold")
        ax.text(k + w / 2, vh * 100 + 1.4, f"{vh*100:.0f}", ha="center", fontsize=11, color=c, fontweight="bold")
    # arrows showing the drop for the two models
    for k, (lab, vo, vh, c) in zip(x, rows):
        if lab.startswith("Skyline"):
            continue
        ax.annotate("", xy=(k + w / 2, vh * 100 + 6), xytext=(k - w / 2, vo * 100 - 3),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.4, alpha=0.6,
                                    connectionstyle="arc3,rad=-0.25"))

    ax.axhline(chance, color=INK, ls=":", lw=1.3)
    ax.text(len(rows) - 0.5, chance + 1.5, f"chance {chance:.0f}%", ha="right", fontsize=9.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=11.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("blind grader accuracy (%)", fontsize=11.5)
    ax.set_title("Specificity reversal — full-explanation suffix prediction\n"
                 "off-document distractors (topic-separable) → same-document (topic controlled)",
                 fontsize=13, fontweight="bold")
    # legend proxies
    ax.bar(0, 0, color=GRAY, alpha=0.32, hatch="///", label="off-document distractors (topic separates the answer)")
    ax.bar(0, 0, color=GRAY, alpha=0.95, label="same-document distractors (must identify the specific continuation)")
    ax.legend(fontsize=9.5, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRAY, alpha=0.22)
    fig.text(0.01, -0.02,
             "Qwen3.6-27B, layer 42, 250 held-out Ultra-FineWeb contexts, Haiku-4.5 grader, 10-way choice. "
             "Same-doc distractors = 9 other 32-token windows from the true answer's own document.",
             fontsize=8.3, color=GRAY, ha="left")
    fig.tight_layout()
    out = HERE / "specificity_reversal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
