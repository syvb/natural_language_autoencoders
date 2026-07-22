"""Position (item index) of hallucinations vs everything else, per model.

Hallucination = model-aware definition (§3g): item still judged unfaithful after
the judge saw the model's own generated continuation. Everything else = all other
judged items.

Outputs:
  results/fig_halluc_mean_index.png        — mean index bar chart, both models
  results/fig_halluc_index_dist_{mat,std}.png — full position distributions
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
H = {"CONTRADICTED", "FABRICATED"}
RED, GREY = "#8a1c13", "#5b6773"
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
withgen = json.load(open(HERE / "results" / "withgen_verdicts.json"))

ARMS = [("mat", "subset_scores_mat.json", "matryoshka (lines)", "line"),
        ("std", "subset_scores_std.json", "standard (sentences)", "sentence")]


def positions(arm, fn):
    """(halluc_positions, else_positions) for one model."""
    ph, pe = [], []
    for e in json.load(open(HERE / "results" / fn))["entries"]:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            for k in range(len(rec["units"])):
                key = f"{arm}|{e['ci']}|{ri}|{k}"
                v = faith.get(key)
                if v is None:
                    continue
                hal = (v in H) and (withgen.get(key) in H)
                (ph if hal else pe).append(k)
    return np.array(ph), np.array(pe)


def mean_index_chart():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ticks, xpos = [], 0
    for arm, fn, name, _ in ARMS:
        a, b = positions(arm, fn)
        p = stats.mannwhitneyu(a, b).pvalue
        for vals, col, lab in [(b, GREY, "everything else"),
                               (a, RED, "hallucination (model, given context)")]:
            v = np.array(vals)
            se = v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(xpos, v.mean(), 0.8, yerr=1.96 * se, capsize=4, color=col,
                   label=lab if arm == "mat" else None)
            ax.text(xpos, v.mean() + 0.14, f"{v.mean():.2f}", ha="center",
                    fontsize=10, fontweight="bold", color=col)
            ax.text(xpos, 0.08, f"n={len(v)}", ha="center", fontsize=7,
                    color="white", rotation=90, va="bottom")
            xpos += 1
        ticks.append((xpos - 1.5, name.replace(" (", "\n(")))
        xpos += 0.8
        print(f"{arm}: halluc mean {a.mean():.2f} vs else {b.mean():.2f} (MWU p={p:.1e})")
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=11)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("mean item index (0-based)", fontsize=11)
    ax.set_title("Average position of hallucinations vs everything else\n"
                 "opposite directions: mat hallucinates LATER, std EARLIER", fontsize=11.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, -0.02,
             "Hallucination = still judged unfaithful after seeing the model's own output (§3g). "
             "Error bars: 95% CI of the mean.\nBoth gaps highly significant "
             "(MWU p=2e-29 mat, p=3e-47 std).",
             fontsize=7.6, color="#777", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "fig_halluc_mean_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[saved] fig_halluc_mean_index.png")


def dist_charts():
    for arm, fn, name, unit in ARMS:
        a, b = positions(arm, fn)
        kmax = int(max(a.max(), b.max()))
        xs = np.arange(kmax + 1)
        fh = np.array([(a == k).mean() for k in xs])
        fe = np.array([(b == k).mean() for k in xs])
        top = max(fh.max(), fe.max())
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        w = 0.42
        ax.bar(xs - w / 2, fe, w, color=GREY, label=f"everything else (n={len(b)})")
        ax.bar(xs + w / 2, fh, w, color=RED,
               label=f"hallucination (model, given context) (n={len(a)})")
        ax.set_ylim(0, top * 1.32)   # headroom so mean labels clear the bars/legend
        # anchor each mean label away from the other line so close means don't collide
        left_first = b.mean() <= a.mean()
        for v, col, ha in [(b.mean(), GREY, "right" if left_first else "left"),
                           (a.mean(), RED, "left" if left_first else "right")]:
            ax.axvline(v, color=col, ls="--", lw=1.4)
            ax.text(v + (0.06 if ha == "left" else -0.06), top * 1.24, f"mean {v:.2f}",
                    color=col, ha=ha, fontsize=8.5, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xlabel(f"item position ({unit} index)", fontsize=11)
        ax.set_ylabel("fraction of group at this position", fontsize=11)
        ax.set_title(f"{name} — position distribution: hallucinations vs everything else",
                     fontsize=11.5)
        ax.legend(fontsize=9, loc="upper right" if arm == "std" else "lower left")
        ax.grid(axis="y", color="#ccc", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(0.02, -0.01,
                 "Each group normalized to sum to 1. Dashed lines = group means. Hallucination = "
                 "still judged unfaithful after seeing\nthe model's own generated output (§3g).",
                 fontsize=7.6, color="#777", ha="left", va="top")
        fig.tight_layout()
        fig.savefig(HERE / "results" / f"fig_halluc_index_dist_{arm}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] fig_halluc_index_dist_{arm}.png")


if __name__ == "__main__":
    mean_index_chart()
    dist_charts()
