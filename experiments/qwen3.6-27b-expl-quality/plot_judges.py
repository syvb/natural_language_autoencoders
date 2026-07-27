"""Judge comparison: mat's share of forced-choice native-format usefulness
judgments, gemma-4-31b vs Fable 5 (both provider-pinned).

    python plot_judges.py   # -> results/fig_judge_compare.png
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GRAY, INK = "#8a8f98", "#22262b"
JCOL = {"gemma-4-31b (CoreWeave)": "#2aa198", "Fable 5 (Vertex, low think)": "#7a52c7"}
GROUPS = ["all", "pretrain", "wildchat"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    man = {c["ci"]: c["domain"] for c in
           json.load(open(HERE / "data" / "manifest_quality.json"))["contexts"]}
    runs = {
        "gemma-4-31b (CoreWeave)":
            json.load(open(HERE / "results" / "transplant_mixed_paired.json")),
        "Fable 5 (Vertex, low think)":
            json.load(open(HERE / "results" / "fable5_native_paired.json"))["paired"],
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    w = 0.34
    for j, (jname, rows) in enumerate(runs.items()):
        for i, g in enumerate(GROUPS):
            cnt = Counter(r["winner"] for r in rows
                          if g == "all" or man[r["ci"]] == g)
            n = cnt["mat"] + cnt["std"]
            p, lo, hi = wilson(cnt["mat"], n)
            x = i + (j - 0.5) * w
            ax.bar(x, p * 100, w * 0.9, color=JCOL[jname], zorder=3)
            ax.errorbar(x, p * 100, [[(p - lo) * 100], [(hi - p) * 100]],
                        color=INK, lw=1.5, capsize=3.5, zorder=4)
            ax.annotate(f"{p*100:.0f}", (x, hi * 100 + 2.5), ha="center",
                        fontsize=10.5, fontweight="bold", color=JCOL[jname])
            ax.annotate(f"n={n}", (x, 3), ha="center", fontsize=8,
                        color="white", fontweight="bold")
    ax.axhline(50, color=INK, ls=":", lw=1.5, zorder=2)
    ax.text(2.44, 51.3, "50% = no preference", ha="right", fontsize=9, color=INK)
    ax.set_xticks(range(3), GROUPS, fontsize=11.5)
    ax.set_ylim(0, 80)
    ax.set_ylabel("matryoshka's share of judgments (%)", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRAY, alpha=0.25, zorder=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l)
                       for l, c in JCOL.items()],
              fontsize=10, loc="upper right", frameon=False)
    ax.set_title("Which explanation better reveals the model's internals?\n"
                 "Forced-choice pairs, NATIVE formats, usefulness-only prompt — "
                 "the verdict is judge-dependent",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02,
             "Same prompt and pairs for both judges; both A/B orders. Fable 5: "
             "100 contexts (16/200 calls returned empty content and are "
             "excluded). Gemma: all 500 contexts.",
             fontsize=8.2, color=GRAY, ha="left", va="top")
    fig.tight_layout()
    out = HERE / "results" / "fig_judge_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
