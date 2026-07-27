"""Figure for the format-transplant / usefulness-only probe.

    python plot_transplant.py   # -> results/fig_transplant_<slug>.png
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK = "#2a78d6", "#e34948", "#8a8f98", "#22262b"
SLUG = "google-gemma-4-31b-it"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    T = json.load(open(HERE / "results" / f"transplant_{SLUG}.json"))
    mixed = json.load(open(HERE / "results" / "transplant_mixed_paired.json"))
    Q = json.load(open(HERE / "results" / f"quality_{SLUG}.json"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    # ── panel A: absolute usefulness, 2 arms x 2 formats ─────────────────────
    ax = axes[0]
    A = {}
    for r in T["absolute"]:
        A.setdefault((r["arm"], r["fmt"]), []).append(r["usefulness"])
    xpos = {("mat", "list"): 0, ("mat", "prose"): 0.32,
            ("std", "prose"): 1.0, ("std", "list"): 1.32}
    for (arm, fmt), x in xpos.items():
        vals = A[(arm, fmt)]
        m = sum(vals) / len(vals)
        se = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 / len(vals) ** 0.5
        native = (arm == "mat") == (fmt == "list")
        col = BLUE if arm == "mat" else RED
        ax.bar(x, m, 0.29, color=col, alpha=1.0 if native else 0.45, zorder=3)
        ax.errorbar(x, m, 1.96 * se, color=INK, lw=1.3, capsize=3, zorder=4)
        ax.annotate(f"{m:.2f}", (x, m + 0.12), ha="center", fontsize=10,
                    fontweight="bold", color=col)
        ax.annotate(fmt + ("\n(native)" if native else "\n(transplant)"),
                    (x, 0.12), ha="center", fontsize=8.2, color="white",
                    fontweight="bold")
    ax.set_xticks([0.16, 1.16], ["matryoshka", "standard"], fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_ylabel("usefulness (1-5), usefulness-only prompt", fontsize=10.5)
    ax.set_title("format barely matters; content gap is large",
                 fontsize=11)

    # ── panel B: mat's pairwise win share across pairing conditions ──────────
    ax = axes[1]
    conds = []
    v = Counter(r["more_useful"] for r in Q["paired"])
    conds.append(("old multi-criteria prompt\nnative formats", v))
    conds.append(("usefulness-only prompt\nnative formats",
                  Counter(r["winner"] for r in mixed)))
    for fmt in ("list", "prose"):
        conds.append((f"usefulness-only prompt\nboth rendered as {fmt}",
                      Counter(r["winner"] for r in T["paired"] if r["fmt"] == fmt)))
    ys = range(len(conds))[::-1]
    for y, (name, v) in zip(ys, conds):
        dec = v["mat"] + v["std"]
        p, lo, hi = wilson(v["mat"], dec)
        ax.barh(y, p * 100, 0.55, color=BLUE, zorder=3)
        ax.barh(y, 100 - p * 100, 0.55, left=p * 100, color=RED, alpha=0.8, zorder=3)
        ax.errorbar(p * 100, y, xerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]],
                    color=INK, lw=1.6, capsize=3, zorder=5)
        ax.annotate(f"{p*100:.0f}%", (p * 100, y + 0.38), ha="center", fontsize=10,
                    fontweight="bold", color=INK)
        ax.text(-2, y, name, ha="right", va="center", fontsize=9.3)
    ax.axvline(50, color=INK, ls=":", lw=1.4, zorder=4)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("matryoshka's share of pairwise judgments (%) — forced choice\n"
                  "(top bar: old run allowed ties; its ~5% ties excluded)",
                  fontsize=10)
    ax.set_title("the pairwise loss was the prompt, not the format",
                 fontsize=11)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y" if a is axes[0] else "x", color=GRAY, alpha=0.25, zorder=0)

    fig.suptitle("Breaking the style confound — usefulness for understanding "
                 "the model's internals\nQwen3.6-27B NLAs, 500 held-out contexts, "
                 "gemma-4-31b judge (CoreWeave-pinned)",
                 fontsize=12.5, fontweight="bold", y=1.06)
    fig.tight_layout()
    out = HERE / "results" / f"fig_transplant_{SLUG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
