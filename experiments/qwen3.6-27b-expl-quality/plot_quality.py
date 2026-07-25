"""Figures for the explanation-quality eval (reads quality_analysis_<slug>.json).

    python plot_quality.py                          # default gemma judge
    python plot_quality.py --judge nex-agi/nex-n2-mini
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"
ARM_COL = {"mat": BLUE, "std": RED}
ARM_NAME = {"mat": "matryoshka", "std": "standard"}
GROUPS = ["all", "pretrain", "wildchat"]
PDIMS = ["overall", "more_useful", "more_grounded", "more_coherent"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="google/gemma-4-31b-it")
    args = ap.parse_args()
    slug = re.sub(r"[^a-z0-9]+", "-", args.judge.lower()).strip("-")
    A = json.load(open(HERE / "results" / f"quality_analysis_{slug}.json"))
    jname = args.judge.split("/")[-1]

    # ── fig 1: calibrated hallucination rates ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    w = 0.35
    for i, g in enumerate(GROUPS):
        for j, arm in enumerate(("mat", "std")):
            b = A["absolute"][g][arm]["halluc_rate_ctx"]
            x = i + (j - 0.5) * w
            ax.bar(x, b["mean"] * 100, w * 0.92, color=ARM_COL[arm], zorder=3)
            ax.errorbar(x, b["mean"] * 100,
                        [[(b["mean"] - b["lo"]) * 100], [(b["hi"] - b["mean"]) * 100]],
                        color=INK, lw=1.4, capsize=3, zorder=4)
            ax.annotate(f"{b['mean']*100:.0f}", (x, b["mean"] * 100 + 3.5),
                        ha="center", fontsize=9.5, fontweight="bold",
                        color=ARM_COL[arm])
    sky = A["absolute"]["all"]["skyline"]["halluc_rate_ctx"]["mean"] * 100
    flo = (A["absolute"]["all"]["floor_mat"]["halluc_rate_ctx"]["mean"] +
           A["absolute"]["all"]["floor_std"]["halluc_rate_ctx"]["mean"]) / 2 * 100
    ax.axhline(sky, color=GRAY, ls="--", lw=1.6)
    ax.text(2.42, sky + 2, f"skyline (passage itself): {sky:.0f}%",
            ha="right", fontsize=8.5, color=GRAY)
    ax.axhline(flo, color=INK, ls="--", lw=1.6, alpha=0.7)
    ax.text(2.42, flo - 6, f"derangement floor (wrong passage): {flo:.0f}%",
            ha="right", fontsize=8.5, color=INK, alpha=0.8)
    ax.set_xticks(range(3), [g + f"\n(n={A['absolute'][g]['mat']['n']}/arm)"
                             for g in GROUPS], fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("hallucinated / checkable claims (%)", fontsize=11)
    ax.set_title("per checkable claim", fontsize=11.5)

    ax = axes[1]
    for i, g in enumerate(GROUPS):
        for j, arm in enumerate(("mat", "std")):
            b = A["absolute"][g][arm]["halluc_per100w"]
            x = i + (j - 0.5) * w
            ax.bar(x, b["mean"], w * 0.92, color=ARM_COL[arm], zorder=3)
            ax.errorbar(x, b["mean"], [[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                        color=INK, lw=1.4, capsize=3, zorder=4)
            ax.annotate(f"{b['mean']:.1f}", (x, b["hi"] + 0.15), ha="center",
                        fontsize=9.5, fontweight="bold", color=ARM_COL[arm])
    ax.set_xticks(range(3), GROUPS, fontsize=10)
    ax.set_ylabel("hallucinated claims per 100 words", fontsize=11)
    ax.set_title("per 100 words of explanation", fontsize=11.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRAY, alpha=0.25, zorder=0)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="matryoshka"),
                        plt.Rectangle((0, 0), 1, 1, color=RED, label="standard")],
               loc="upper center", ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Claim-level hallucination, calibrated — {jname} judge, "
                 "500 held-out contexts", fontsize=12.5, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_halluc_{slug}.png", dpi=150,
                bbox_inches="tight")
    print("[saved] fig_halluc")

    # ── fig 2: paired preferences ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, g in zip(axes, GROUPS):
        P = A["paired"][g]
        ys = range(len(PDIMS))[::-1]
        for y, dim in zip(ys, PDIMS):
            v = P[dim]["votes"]
            tot = sum(v.values())
            m, t = v.get("mat", 0) / tot * 100, v.get("TIE", 0) / tot * 100
            s = v.get("std", 0) / tot * 100
            ax.barh(y, m, color=BLUE, zorder=3)
            ax.barh(y, t, left=m, color=GRAY, alpha=0.55, zorder=3)
            ax.barh(y, s, left=m + t, color=RED, zorder=3)
            ax.text(m / 2, y, f"{m:.0f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
            ax.text(m + t + s / 2, y, f"{s:.0f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
            ax.text(101.5, y, f"flip {P[dim]['flip_rate']:.0%}", va="center",
                    fontsize=8, color=GRAY)
        ax.set_yticks(list(ys), [d.replace("more_", "") for d in PDIMS], fontsize=10)
        ax.axvline(50, color=INK, ls=":", lw=1.2, zorder=4)
        ax.set_xlim(0, 100)
        ax.set_title(g, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("% of judgments", fontsize=10)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="matryoshka preferred"),
                        plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.55, label="tie"),
                        plt.Rectangle((0, 0), 1, 1, color=RED, label="standard preferred")],
               loc="upper center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Head-to-head preference (both A/B orders) — {jname} judge",
                 fontsize=12.5, fontweight="bold", y=1.09)
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_paired_{slug}.png", dpi=150,
                bbox_inches="tight")
    print("[saved] fig_paired")

    # ── fig 3: rubric scores with calibration anchors ────────────────────────
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    dims = ["coherence", "usefulness", "informativeness"]
    for i, dim in enumerate(dims):
        for j, arm in enumerate(("mat", "std")):
            b = A["absolute"]["all"][arm][dim]
            x = i + (j - 0.5) * 0.22
            ax.errorbar(x, b["mean"], [[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                        fmt="o", ms=9, color=ARM_COL[arm], lw=2, capsize=4, zorder=4)
            ax.annotate(f"{b['mean']:.2f}", (x, b["mean"]),
                        textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=9.5, color=ARM_COL[arm], fontweight="bold")
        for leg, mk in (("skyline", "^"), ("floor_mat", "v"), ("floor_std", "v")):
            b = A["absolute"]["all"][leg][dim]
            ax.scatter(i, b["mean"], marker=mk, s=55, color=GRAY, zorder=3)
    ax.set_xticks(range(len(dims)), dims, fontsize=11)
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks(range(1, 6))
    ax.set_ylabel("judge score (1-5)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRAY, alpha=0.25, zorder=0)
    ax.set_title(f"Rubric scores with calibration anchors — {jname} judge\n"
                 "▲ skyline (passage itself)   ▼ derangement floors (wrong passage)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE,
                                  label="matryoshka"),
                       plt.Line2D([], [], marker="o", ls="", ms=8, color=RED,
                                  label="standard")],
              fontsize=10, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_scores_{slug}.png", dpi=150,
                bbox_inches="tight")
    print("[saved] fig_scores")


if __name__ == "__main__":
    main()
