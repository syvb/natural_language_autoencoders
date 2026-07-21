"""Are hallucinations lower marginal FVE under the BETTER labels? Within-position
gap (hallucinated - SUPPORTED) for three hallucination definitions:
  corpus loose       — single judge vs corpus continuation
  corpus 2-model     — nex + gpt-4o-mini both confirm a concrete fabrication (§3f)
  model-aware        — judge also sees the model's own generated output (§3g);
                       hallucinated = residual, faithful = SUPPORTED + flipped
Negative coef = hallucinations reconstruct worse (the hypothesis). ~0 = no signal.
"""
import json
import random
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
H = {"CONTRADICTED", "FABRICATED"}
MODELS = ["nex-agi/nex-n2-mini", "openai/gpt-4o-mini"]
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
strict = json.load(open(HERE / "results" / "strict_halluc.json"))
withgen = json.load(open(HERE / "results" / "withgen_verdicts.json"))


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def load(arm, fn):
    out = []
    for e in json.load(open(HERE / "results" / fn))["entries"]:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            m = marg(rec)
            for k in range(len(m)):
                key = f"{arm}|{e['ci']}|{ri}|{k}"
                v = faith.get(key)
                if v is not None:
                    out.append(dict(key=key, pos=k, mg=m[k], v=v))
    return out


def ols_ci(sup, hal):
    rows = [(r["pos"], r["mg"], 0) for r in sup] + [(r["pos"], r["mg"], 1) for r in hal]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), se, float(2 * stats.t.sf(abs(b[-1] / se), dof))


DEFS = ["corpus\nloose", "corpus\n2-model strict", "model-aware\n(sees model output)"]
COLS = ["#e34948", "#8a6d3b", "#2a78d6"]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
for ax, (arm, fn, title) in zip(axes, [("mat", "subset_scores_mat.json", "matryoshka (lines)"),
                                       ("std", "subset_scores_std.json", "standard (sentences)")]):
    R = load(arm, fn)
    sup0 = [r for r in R if r["v"] == "SUPPORTED"]
    loose = [r for r in R if r["v"] in H]
    defs = [
        (sup0, loose),
        (sup0, [r for r in loose if all(strict[r["key"]].values())]),
        (sup0 + [r for r in loose if withgen.get(r["key"]) == "SUPPORTED"],
         [r for r in loose if withgen.get(r["key"]) in H]),
    ]
    ys = [2, 1, 0]
    for y, (sup, hal), col, name in zip(ys, defs, COLS, DEFS):
        b, se, p = ols_ci(sup, hal)
        ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color=col, ms=8, capsize=4, lw=2)
        sig = "*" if p < 0.05 else ""
        ax.text(b, y + 0.16, f"{b:+.4f}{sig} (p={p:.2f}, n={len(hal)})", ha="center", fontsize=8.5, color=col)
    ax.axvline(0, color="#444", lw=1.1, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels(DEFS, fontsize=9.5)
    ax.set_xlabel("within-position marginal FVE:  hallucinated − SUPPORTED\n(◀ negative = hallucinations reconstruct worse)", fontsize=9.5)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", color="#ccc", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.25)
fig.suptitle("Do hallucinations have lower marginal FVE? The clean definition says ≈0;\n"
             "the model-aware labels hint 'yes' for standard (−0.04*) — but that is partly a relabeling artifact",
             fontsize=12, y=1.09)
fig.subplots_adjust(bottom=0.30)
fig.text(0.5, -0.10,
         "2-model strict is the confound-free definition (purifies on faithfulness only, via nex+gpt-4o-mini consensus) and shows ≈0 for both models.\n"
         "The model-aware definition moves the high-marginal 'correct-prediction' items OUT of the hallucinated group and INTO faithful — and since "
         "matching the\nmodel's own output correlates with being reconstructable, that shift mechanically widens the gap. So the −0.04 (std) is suggestive but "
         "not clean causal\nevidence; the cleanest cut remains ≈0. Error bars: 95% CI; * = p<0.05.",
         fontsize=7.7, color="#777", ha="center", va="top")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(HERE / "results" / "fig_fve_by_def.png", dpi=150, bbox_inches="tight")
print("[saved] results/fig_fve_by_def.png")
