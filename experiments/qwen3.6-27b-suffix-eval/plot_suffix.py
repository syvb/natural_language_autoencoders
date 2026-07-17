"""Plot the Suffix-Prediction eval (two panels, no GPU):

  A. Validity + headline: skyline (items solvable), both models' blind accuracy
     with Wilson CIs, and the shuffled-pairing controls sitting on the chance
     line — proof the ~99% is real signal, not a format cue.
  B. Budget curve: accuracy vs. how many explanation LINES the grader may read.
     The front-loading test — how fast each model reaches ceiling.

    python plot_suffix.py    # results/results.json (+ results/budget.json) -> suffix_eval.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"


def main():
    r = json.load(open(HERE / "results" / "results.json"))
    arms = r["arms"]
    chance = 100.0 / r["meta"]["n_options"]
    bpath = HERE / "results" / "budget.json"
    budget = json.load(open(bpath)) if bpath.exists() else None

    ncols = 2 if budget else 1
    fig, axes = plt.subplots(1, ncols, figsize=(15 if budget else 9.2, 5.0))
    axA = axes[0] if budget else axes

    # ── Panel A: validity + headline bars ─────────────────────────────────────
    rows = [("Skyline (passage → answer)", r["skyline"]["acc"], r["skyline"]["ci"], GRAY, False)]
    if "mat" in arms:
        rows.append(("Matryoshka — primary", arms["mat"]["primary"]["acc"], arms["mat"]["primary"]["ci"], BLUE, False))
    if "std" in arms:
        rows.append(("Standard — primary", arms["std"]["primary"]["acc"], arms["std"]["primary"]["ci"], RED, False))
    if "mat" in arms:
        rows.append(("Matryoshka — shuffled ctrl", arms["mat"]["shuffled"]["acc"], arms["mat"]["shuffled"]["ci"], BLUE, True))
    if "std" in arms:
        rows.append(("Standard — shuffled ctrl", arms["std"]["shuffled"]["acc"], arms["std"]["shuffled"]["ci"], RED, True))
    ypos = list(range(len(rows)))[::-1]
    for y, (lab, v, ci, c, h) in zip(ypos, rows):
        v100, lo, hi = v * 100, ci[0] * 100, ci[1] * 100
        axA.barh(y, v100, height=0.6, color=c, alpha=0.32 if h else 0.92,
                 hatch="///" if h else None, edgecolor=c, linewidth=1.2, zorder=3)
        axA.plot([lo, hi], [y, y], color=INK, lw=1.5, zorder=4)
        for cap in (lo, hi):
            axA.plot([cap, cap], [y - 0.11, y + 0.11], color=INK, lw=1.5, zorder=4)
        axA.text(min(v100 + 1.6, 101), y, f"{v100:.1f}%", va="center", ha="left",
                 fontsize=10.5, color=INK, fontweight="bold", zorder=5)
    axA.axvline(chance, color=INK, ls=":", lw=1.3, zorder=2)
    axA.text(chance + 0.5, len(rows) - 0.4, f"chance {chance:.0f}%", color=INK, fontsize=9, va="top")
    axA.set_yticks(ypos)
    axA.set_yticklabels([x[0] for x in rows], fontsize=10.5)
    axA.set_xlim(0, 108)
    axA.set_xlabel("blind grader accuracy (%)", fontsize=10.5)
    axA.set_title("A. Identify the true continuation among 10 options (9 off-document)", fontsize=11.5, fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="x", color=GRAY, alpha=0.25, zorder=0)

    # ── Panel B: budget curve ─────────────────────────────────────────────────
    if budget:
        axB = axes[1]
        for tag, c, lab in [("mat", BLUE, "Matryoshka"), ("std", RED, "Standard")]:
            if tag not in budget["arms"]:
                continue
            curve = budget["arms"][tag]["curve"]
            ks = sorted(int(k) for k in curve)
            ys = [curve[str(k)]["acc"] * 100 for k in ks]
            los = [curve[str(k)]["ci"][0] * 100 for k in ks]
            his = [curve[str(k)]["ci"][1] * 100 for k in ks]
            axB.fill_between(ks, los, his, color=c, alpha=0.13, zorder=2)
            axB.plot(ks, ys, "o-", color=c, lw=2.4, ms=7, label=lab, zorder=3)
            for k, y in zip(ks, ys):
                axB.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                             xytext=(0, 8 if tag == "mat" else -14), ha="center",
                             fontsize=8.5, color=c, fontweight="bold")
        axB.axhline(chance, color=INK, ls=":", lw=1.3, zorder=1)
        axB.text(budget["budgets"][0], chance + 1.5, f"chance {chance:.0f}%",
                 color=INK, fontsize=9, ha="left", va="bottom")
        axB.set_xlabel("explanation content tokens the grader may read (first N)", fontsize=10.5)
        axB.set_ylabel("blind grader accuracy (%)", fontsize=10.5)
        axB.set_title("B. How few tokens are enough? (front-loading, U[1,120] train range)",
                      fontsize=11.5, fontweight="bold")
        axB.set_xscale("log")
        axB.set_xticks(budget["budgets"])
        axB.get_xaxis().set_major_formatter(ScalarFormatter())
        axB.set_ylim(0, 105)
        axB.legend(fontsize=10, loc="lower right")
        axB.spines[["top", "right"]].set_visible(False)
        axB.grid(color=GRAY, alpha=0.22, zorder=0)

    n, ro = r["meta"]["n_contexts"], r["meta"]["rollouts"]
    fig.suptitle(f"NLA Suffix-Prediction eval — Qwen3.6-27B, layer 42 | "
                 f"{n} held-out Ultra-FineWeb contexts × {ro} rollouts, Haiku-4.5 grader",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = HERE / "suffix_eval.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")

    print(f"\nskyline {r['skyline']['acc']:.1%} | chance {chance:.0f}%")
    for k, a in arms.items():
        print(f"{k}: primary {a['primary']['acc']:.1%} maj-vote {a['majority_vote']['acc']:.1%} "
              f"shuffled {a['shuffled']['acc']:.1%} agree {a['inter_rollout_agreement']:.0%}")
    if budget:
        for tag, arm in budget["arms"].items():
            curve = arm["curve"]
            pts = ", ".join(f"T{k}:{curve[k]['acc']:.0%}" for k in sorted(curve, key=int))
            print(f"budget {tag} (median {arm['expl_token_len_median']} tok): {pts}")


if __name__ == "__main__":
    main()
