"""Specificity test figure: off-document vs same-document distractors.

Panel A: full-explanation accuracy (skyline / matryoshka / standard) under each
distractor regime — how far the topic-separable ~99% falls when every option is
on-topic.
Panel B: budget curves under same-document distractors (solid) with the
off-document curves (faded dashed) for reference — does the matryoshka's
front-loading advantage survive when topic can't be used?

    python plot_hard.py   # results/{results,results_hard,budget,budget_hard}.json -> specificity.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"


def L(name):
    return json.load(open(HERE / "results" / name))


def main():
    off, hard = L("results.json"), L("results_hard.json")
    boff, bhard = L("budget.json"), L("budget_hard.json")
    chance = 100.0 / off["meta"]["n_options"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.2))

    # ── Panel A: full-length, two regimes ─────────────────────────────────────
    rows = [("Skyline", off["skyline"]["acc"], hard["skyline"]["acc"], GRAY),
            ("Matryoshka", off["arms"]["mat"]["primary"]["acc"], hard["arms"]["mat"]["primary"]["acc"], BLUE),
            ("Standard", off["arms"]["std"]["primary"]["acc"], hard["arms"]["std"]["primary"]["acc"], RED)]
    x = range(len(rows))
    w = 0.38
    for k, (lab, vo, vh, c) in zip(x, rows):
        axA.bar(k - w / 2, vo * 100, w, color=c, alpha=0.35, hatch="///", edgecolor=c, linewidth=1.1, zorder=3)
        axA.bar(k + w / 2, vh * 100, w, color=c, alpha=0.95, edgecolor=c, linewidth=1.1, zorder=3)
        axA.text(k - w / 2, vo * 100 + 1.5, f"{vo*100:.0f}", ha="center", fontsize=9.5, color=c, fontweight="bold")
        axA.text(k + w / 2, vh * 100 + 1.5, f"{vh*100:.0f}", ha="center", fontsize=9.5, color=c, fontweight="bold")
    axA.axhline(chance, color=INK, ls=":", lw=1.3)
    axA.text(len(rows) - 0.5, chance + 1.5, f"chance {chance:.0f}%", ha="right", fontsize=9, color=INK)
    axA.set_xticks(list(x))
    axA.set_xticklabels([r[0] for r in rows], fontsize=11)
    axA.set_ylim(0, 105)
    axA.set_ylabel("blind grader accuracy (%)", fontsize=10.5)
    axA.set_title("A. Full explanation — topic vs. specificity", fontsize=11.5, fontweight="bold")
    # legend proxies
    axA.bar(0, 0, color=GRAY, alpha=0.35, hatch="///", label="off-document distractors (topic-separable)")
    axA.bar(0, 0, color=GRAY, alpha=0.95, label="same-document distractors (specificity)")
    axA.legend(fontsize=9, loc="lower left")
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="y", color=GRAY, alpha=0.22)

    # ── Panel B: budget curves, same-doc solid + off-doc faded ────────────────
    def curve(b, tag):
        c = b["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        return ks, [c[str(k)]["acc"] * 100 for k in ks], \
            [c[str(k)]["ci"][0] * 100 for k in ks], [c[str(k)]["ci"][1] * 100 for k in ks]

    hard_curves = {tag: curve(bhard, tag) for tag in ("mat", "std")}
    for tag, c, lab in [("mat", BLUE, "Matryoshka"), ("std", RED, "Standard")]:
        ks, yo, _, _ = curve(boff, tag)
        axB.plot(ks, yo, "--", color=c, lw=1.6, alpha=0.4, zorder=2)
        ks, yh, lo, hi = hard_curves[tag]
        axB.fill_between(ks, lo, hi, color=c, alpha=0.13, zorder=2)
        axB.plot(ks, yh, "o-", color=c, lw=2.6, ms=7, label=f"{lab} (same-doc)", zorder=4)
        other = hard_curves["std" if tag == "mat" else "mat"][1]
        for i, (k, y) in enumerate(zip(ks, yh)):
            above = y >= other[i]           # label on the outside of the pair
            axB.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                         xytext=(0, 9 if above else -16), ha="center",
                         fontsize=8.5, color=c, fontweight="bold")
    axB.axhline(chance, color=INK, ls=":", lw=1.3)
    axB.text(boff["budgets"][2], chance + 1.5, f"chance {chance:.0f}%", ha="center",
             fontsize=9, color=INK)
    # skyline-hard reference
    sk = hard["skyline"]["acc"] * 100
    axB.axhline(sk, color=GRAY, ls="-.", lw=1.3, alpha=0.8)
    axB.text(boff["budgets"][-1], sk - 4, f"same-doc skyline {sk:.0f}%", ha="right", fontsize=8.5, color=GRAY)
    axB.set_xscale("log")
    axB.set_xticks(boff["budgets"])
    axB.get_xaxis().set_major_formatter(ScalarFormatter())
    axB.set_ylim(0, 105)
    axB.set_xlabel("explanation content tokens the grader may read (first N)", fontsize=10.5)
    axB.set_ylabel("blind grader accuracy (%)", fontsize=10.5)
    axB.set_title("B. Front-loading under same-document distractors\n(dashed = off-document, for reference)",
                  fontsize=11.5, fontweight="bold")
    axB.legend(fontsize=9.5, loc="center right")
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(color=GRAY, alpha=0.22)

    n = off["meta"]["n_contexts"]
    fig.suptitle(f"Specificity test: does the front-loading survive when topic can't be used? "
                 f"— Qwen3.6-27B L42, {n} contexts, Haiku grader",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = HERE / "specificity.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")

    print(f"\nskyline off {off['skyline']['acc']:.1%} -> same-doc {hard['skyline']['acc']:.1%}")
    for m in ["mat", "std"]:
        print(f"{m}: full off {off['arms'][m]['primary']['acc']:.1%} -> same-doc {hard['arms'][m]['primary']['acc']:.1%}")
    for m in ["mat", "std"]:
        c = bhard["arms"][m]["curve"]
        print(f"budget same-doc {m}:", {int(k): round(c[k]['acc'] * 100, 1) for k in sorted(c, key=int)})


if __name__ == "__main__":
    main()
