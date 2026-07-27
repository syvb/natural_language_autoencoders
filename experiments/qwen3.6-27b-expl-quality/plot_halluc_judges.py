"""Which arm has MORE hallucinations — Fable 5 holistic pairwise vs gemma's
claim-count comparison (independent judges + methods, same explanations).

    python plot_halluc_judges.py   # -> results/fig_halluc_compare.png
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
COLS = {"Fable 5 pairwise (Vertex, low think)": "#7a52c7",
        "gemma-4-31b claim counts (CoreWeave)": "#2aa198"}
GROUPS = ["all", "pretrain", "wildchat"]
HAL = {"UNSUPPORTED", "CONTRADICTED"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    man = {c["ci"]: c["domain"] for c in
           json.load(open(HERE / "data" / "manifest_quality.json"))["contexts"]}
    fab = json.load(open(HERE / "results" / "fable5_halluc_paired.json"))["paired"]
    Q = json.load(open(HERE / "results" / "quality_google-gemma-4-31b-it.json"))
    cnt = {}
    for r in Q["absolute"]:
        if r["arm"] in ("mat", "std"):
            cnt[(r["arm"], r["ci"])] = sum(1 for c in r["claims"]
                                           if c["label"] in HAL)

    def series(g):
        v = Counter(r["more_halluc"] for r in fab
                    if g == "all" or man[r["ci"]] == g)
        fk, fn = v["mat"], v["mat"] + v["std"]
        m = s = 0
        for ci in man:
            if g != "all" and man[ci] != g:
                continue
            a, b = cnt.get(("mat", ci)), cnt.get(("std", ci))
            if a is None or b is None or a == b:
                continue
            m, s = (m + 1, s) if a > b else (m, s + 1)
        return (fk, fn), (m, m + s)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    w = 0.34
    for j, name in enumerate(COLS):
        for i, g in enumerate(GROUPS):
            (fk, fn), (gk, gn) = series(g)
            k, n = (fk, fn) if j == 0 else (gk, gn)
            p, lo, hi = wilson(k, n)
            x = i + (j - 0.5) * w
            ax.bar(x, p * 100, w * 0.9, color=list(COLS.values())[j], zorder=3)
            ax.errorbar(x, p * 100, [[(p - lo) * 100], [(hi - p) * 100]],
                        color=INK, lw=1.5, capsize=3.5, zorder=4)
            ax.annotate(f"{p*100:.0f}", (x, hi * 100 + 2.5), ha="center",
                        fontsize=10.5, fontweight="bold",
                        color=list(COLS.values())[j])
            ax.annotate(f"n={n}", (x, 3), ha="center", fontsize=8,
                        color="white", fontweight="bold")
    ax.axhline(50, color=INK, ls=":", lw=1.5, zorder=2)
    ax.text(2.44, 45.5, "50% = no difference", ha="right", fontsize=9, color=INK)
    ax.set_xticks(range(3), GROUPS, fontsize=11.5)
    ax.set_ylim(0, 92)
    ax.set_ylabel("MATRYOSHKA named as having more hallucinations (%)",
                  fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRAY, alpha=0.25, zorder=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l)
                       for l, c in COLS.items()],
              fontsize=9.5, loc="upper right", frameon=False)
    ax.set_title("Which explanation contains MORE hallucinations?\n"
                 "Two independent judges/methods agree: the matryoshka — by "
                 "count, not rate", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02,
             "Fable 5: forced-choice pairs, native formats, both A/B orders, 100 "
             "contexts (17/200 empty replies excluded), order-flip 11%.\n"
             "gemma: per-context comparison of hallucinated-claim COUNTS from the "
             "absolute claim-labeling leg (ties excluded). Per-claim RATE is a "
             "tie (47.0% vs 47.5%) — the matryoshka simply asserts more claims.",
             fontsize=8.2, color=GRAY, ha="left", va="top")
    fig.tight_layout()
    out = HERE / "results" / "fig_halluc_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
