"""Final-token-revealed suffix eval under BOTH judges (Haiku 4.5 | nex-n2-mini).

Per panel: solid = explanation + final token; faded dashed = explanation only
(that judge's own prior run); gray line = final token alone for that judge.

    python plot_ft_graders.py   # -> ft_graders.png
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
    p = HERE / "results" / name
    return json.load(open(p)) if p.exists() else None


def panel(ax, ft, noft, title):
    tko = ft["token_only"]["acc"] * 100
    for tag, col in [("mat", BLUE), ("std", RED)]:
        if noft and tag in noft["arms"]:
            c = noft["arms"][tag]["curve"]
            ks = sorted(int(k) for k in c)
            ax.plot(ks, [c[str(k)]["acc"] * 100 for k in ks], "--", color=col,
                    lw=1.6, alpha=0.35, zorder=2)
        c = ft["arms"][tag]["curve"]
        ks = sorted(int(k) for k in c)
        ys = [c[str(k)]["acc"] * 100 for k in ks]
        lo = [c[str(k)]["ci"][0] * 100 for k in ks]
        hi = [c[str(k)]["ci"][1] * 100 for k in ks]
        ax.fill_between(ks, lo, hi, color=col, alpha=0.12, zorder=2)
        ax.plot(ks, ys, "o-", color=col, lw=2.6, ms=6.5, zorder=4)
        for k, y in zip(ks, ys):
            ax.annotate(f"{y:.0f}", (k, y), textcoords="offset points",
                        xytext=(0, 9 if tag == "mat" else -16), ha="center",
                        fontsize=8.5, color=col, fontweight="bold", zorder=5)
    ax.axhline(tko, color=GRAY, ls="-", lw=2.0, alpha=0.75, zorder=3)
    ax.text(ks[-1], tko + 1.5, f"token alone {tko:.0f}%", ha="right",
            fontsize=9, color=GRAY, fontweight="bold")
    ax.axhline(10, color=INK, ls=":", lw=1.2, zorder=1)
    ax.text(4, 11, "chance 10%", ha="left", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(ks[0] * 0.88, ks[-1] * 1.12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("explanation content tokens (first N)", fontsize=10.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRAY, alpha=0.22, zorder=0)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.6, 5.6), sharey=True)
    panel(axA, L("ft_results.json"), L("budget_hard.json"), "Haiku 4.5 judge")
    panel(axB, L("ft_results_nex.json"), L("budget_hard_nex.json"), "nex-n2-mini judge")
    axA.set_ylabel("blind judge accuracy (%)", fontsize=11)
    axB.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o", label="Matryoshka + final token"),
        Line2D([], [], color=RED, lw=2.6, marker="o", label="Standard + final token"),
        Line2D([], [], color=INK, lw=1.6, ls="--", alpha=0.4, label="explanation only"),
        Line2D([], [], color=GRAY, lw=2.0, alpha=0.75, label="final token only"),
    ], fontsize=9, loc="lower right")
    fig.suptitle("Final token revealed to the judge — grader comparison "
                 "(same-document suffix prediction, Qwen3.6-27B L42)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = HERE / "ft_graders.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"[saved] {out}")

    for name in ("ft_results.json", "ft_results_nex.json"):
        r = L(name)
        if not r:
            continue
        g = "haiku" if "nex" not in name else "nex"
        t4m = r["arms"]["mat"]["curve"].get("4", {}).get("acc", 0) * 100
        t4s = r["arms"]["std"]["curve"].get("4", {}).get("acc", 0) * 100
        print(f"{g}: token-only {r['token_only']['acc']:.1%} | T4 mat {t4m:.0f} std {t4s:.0f} "
              f"(gap {t4m-t4s:+.0f}) | full mat {r['arms']['mat']['full']['acc']:.1%} "
              f"std {r['arms']['std']['full']['acc']:.1%}")


if __name__ == "__main__":
    main()
