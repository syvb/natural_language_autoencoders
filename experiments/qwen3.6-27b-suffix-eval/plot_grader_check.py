"""Grader-robustness check: Haiku 4.5 vs nex-n2-mini on the same-document eval.

Same items, same explanations, two independent graders. If the conclusions are
grader-robust, the curves should share their shape (matryoshka high-early/flat,
standard low-early/steep, crossover near the training-range top) even if
absolute levels shift.

    python plot_grader_check.py  # -> grader_check.png
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


def L(name):
    return json.load(open(HERE / "results" / name))


def main():
    h_full, n_full = L("results_hard.json"), L("results_hard_nex.json")
    h_bud, n_bud = L("budget_hard.json"), L("budget_hard_nex.json")
    chance = 100.0 / h_full["meta"]["n_options"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.4))

    # ── Panel A: full-explanation bars, grouped by grader ────────────────────
    rows = [("Skyline", lambda r: r["skyline"]["acc"], GRAY),
            ("Matryoshka", lambda r: r["arms"]["mat"]["primary"]["acc"], BLUE),
            ("Standard", lambda r: r["arms"]["std"]["primary"]["acc"], RED)]
    w = 0.36
    for k, (lab, get, c) in enumerate(rows):
        vh, vn = get(h_full) * 100, get(n_full) * 100
        axA.bar(k - w / 2, vh, w, color=c, alpha=0.92, edgecolor=c, zorder=3)
        axA.bar(k + w / 2, vn, w, color=c, alpha=0.45, edgecolor=c, hatch="..", zorder=3)
        axA.text(k - w / 2, vh + 1.4, f"{vh:.0f}", ha="center", fontsize=10.5, color=c, fontweight="bold")
        axA.text(k + w / 2, vn + 1.4, f"{vn:.0f}", ha="center", fontsize=10.5, color=c, fontweight="bold")
    axA.axhline(chance, color=INK, ls=":", lw=1.3)
    axA.text(len(rows) - 0.55, chance + 1.5, f"chance {chance:.0f}%", ha="right", fontsize=9.5, color=INK)
    axA.set_xticks(range(len(rows)))
    axA.set_xticklabels([r[0] for r in rows], fontsize=11.5)
    axA.set_ylim(0, 105)
    axA.set_ylabel("blind grader accuracy (%)", fontsize=11)
    axA.set_title("A. Full explanation, same-document distractors", fontsize=11.5, fontweight="bold")
    axA.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=GRAY, alpha=0.92, ec=GRAY, label="Claude Haiku 4.5 grader"),
        plt.Rectangle((0, 0), 1, 1, fc=GRAY, alpha=0.45, ec=GRAY, hatch="..", label="nex-n2-mini grader"),
    ], fontsize=9.5, loc="lower right")
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="y", color=GRAY, alpha=0.22)

    # ── Panel B: budget curves, linestyle = grader ────────────────────────────
    for b, ls, mk, alpha in [(h_bud, "-", "o", 1.0), (n_bud, "--", "s", 0.75)]:
        for tag, c in [("mat", BLUE), ("std", RED)]:
            cur = b["arms"][tag]["curve"]
            ks = sorted(int(k) for k in cur)
            ys = [cur[str(k)]["acc"] * 100 for k in ks]
            axB.plot(ks, ys, ls, marker=mk, color=c, lw=2.3, ms=6, alpha=alpha, zorder=4)
    axB.axhline(chance, color=INK, ls=":", lw=1.3)
    axB.text(16, chance + 1.6, f"chance {chance:.0f}%", ha="center", fontsize=9.5, color=INK)
    axB.axvline(120, color=GRAY, ls=":", lw=1.2, alpha=0.8)
    axB.text(120, 96, " U[1,120] top", ha="left", fontsize=8.5, color=GRAY)
    ks = sorted(int(k) for k in h_bud["arms"]["mat"]["curve"])
    axB.set_xscale("log")
    axB.set_xticks(ks)
    axB.get_xaxis().set_major_formatter(ScalarFormatter())
    axB.set_ylim(0, 100)
    axB.set_xlabel("explanation content tokens the grader may read (first N)", fontsize=11)
    axB.set_ylabel("blind grader accuracy (%)", fontsize=11)
    axB.set_title("B. Budget curves under both graders", fontsize=11.5, fontweight="bold")
    axB.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.5, label="Matryoshka"),
        Line2D([], [], color=RED, lw=2.5, label="Standard"),
        Line2D([], [], color=INK, lw=2, ls="-", marker="o", label="Haiku 4.5"),
        Line2D([], [], color=INK, lw=2, ls="--", marker="s", alpha=0.75, label="nex-n2-mini"),
    ], fontsize=9.5, loc="center right")
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(color=GRAY, alpha=0.22)

    fig.suptitle("Grader-robustness check — same-document specificity eval, two independent graders",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = HERE / "grader_check.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")

    # console deltas
    for lab, get in [("skyline", lambda r: r["skyline"]["acc"]),
                     ("mat full", lambda r: r["arms"]["mat"]["primary"]["acc"]),
                     ("std full", lambda r: r["arms"]["std"]["primary"]["acc"])]:
        print(f"{lab}: haiku {get(h_full):.1%} vs nex {get(n_full):.1%}")
    for tag in ("mat", "std"):
        hc, nc = h_bud["arms"][tag]["curve"], n_bud["arms"][tag]["curve"]
        shared = sorted(set(int(x) for x in hc) & set(int(x) for x in nc))
        print(f"budget {tag} (haiku/nex): " + "  ".join(
            f"T{k}: {hc[str(k)]['acc']:.0%}/{nc[str(k)]['acc']:.0%}" for k in shared))


if __name__ == "__main__":
    main()
