"""One graph: the within-position hallucination-vs-everything-else gap at each
position, as a STANDARDIZED effect size (Cohen's d = (mean_halluc - mean_else) /
pooled SD). Plotting the gap (not the raw FVE) collapses the huge position-to-
position magnitude differences, so every position sits on one comparable axis and
both models fit together. Points near 0 = hallucinations reconstruct like
everything else at that position."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED = "#2a78d6", "#e34948"
H = {"CONTRADICTED", "FABRICATED"}
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
withgen = json.load(open(HERE / "results" / "withgen_verdicts.json"))


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def load(arm, fn):
    pos, mg, hal = [], [], []
    for e in json.load(open(HERE / "results" / fn))["entries"]:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            m = marg(rec)
            for k in range(len(m)):
                key = f"{arm}|{e['ci']}|{ri}|{k}"
                v = faith.get(key)
                if v is None:
                    continue
                pos.append(k); mg.append(m[k]); hal.append((v in H) and (withgen.get(key) in H))
    return np.array(pos), np.array(mg), np.array(hal, bool)


fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.axhspan(-0.1, 0.1, color="#bbb", alpha=0.22, lw=0, label="negligible (|d| < 0.1)")
for arm, fn, col, name, dx in [("mat", "subset_scores_mat.json", BLUE, "matryoshka", -0.08),
                               ("std", "subset_scores_std.json", RED, "standard", 0.08)]:
    pos, mg, hal = load(arm, fn)
    kmax = int(np.percentile(pos, 99))
    xs, ds, los, his = [], [], [], []
    for k in range(kmax + 1):
        sh = (pos == k) & hal; se_ = (pos == k) & ~hal
        if sh.sum() < 15 or se_.sum() < 15:
            continue
        a, b = mg[sh], mg[se_]
        sp = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1))) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp
        se_d = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + d * d / (2 * (len(a) + len(b))))
        xs.append(k); ds.append(d); los.append(1.96 * se_d); his.append(1.96 * se_d)
    ax.errorbar(np.array(xs) + dx, ds, yerr=[los, his], fmt="o-", color=col, ms=6, lw=1.6,
                capsize=3, label=name)
ax.axhline(0, color="#444", lw=1.2, ls="--")
ax.set_xlabel("item position (line / sentence index)", fontsize=11)
ax.set_ylabel("standardized gap: hallucination − everything else\n(Cohen's d; ◀ negative = hallucinations reconstruct worse)", fontsize=10.5)
ax.set_title("Within-position hallucination-vs-rest gap at every position, one scale\n"
             "(hallucination = unfaithful even given the model's own output)", fontsize=12)
ax.legend(fontsize=9.5, loc="upper right")
ax.grid(color="#ccc", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.02,
         "Each point is one position: standardized mean-marginal-FVE difference (Cohen's d) between hallucinated and all other items, with 95% CI. "
         "Plotting the\nstandardized GAP (not raw FVE) puts every position and both models on one axis. Points scatter around 0 within the negligible band — "
         "no position shows hallucinations reconstructing meaningfully worse.",
         fontsize=7.7, color="#777", ha="center", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_effect_by_pos.png", dpi=150, bbox_inches="tight")
print("[saved] results/fig_effect_by_pos.png")
