"""Simplest cut: marginal FVE by position for "hallucination according to the
model given context" (item still judged hallucinated after the judge saw the
model's OWN generated continuation, §3g) vs "everything else" (all other items)."""
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RED, GREY = "#8a1c13", "#5b6773"
H = {"CONTRADICTED", "FABRICATED"}
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
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
                if v is None:
                    continue
                # hallucination given the model's own context: originally flagged AND still flagged with model output
                hal = (v in H) and (withgen.get(key) in H)
                out.append((k, m[k], hal))
    return out


def ols(rows):
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), float(2 * stats.t.sf(abs(b[-1] / se), dof))


for arm, fn, title, unit in [("mat", "subset_scores_mat.json", "matryoshka (lines)", "line"),
                             ("std", "subset_scores_std.json", "standard (sentences)", "sentence")]:
    R = load(arm, fn)
    coef, p = ols(R)
    ks = np.array([r[0] for r in R]); kmax = int(np.percentile(ks, 99)); xs = list(range(kmax + 1))
    def curve(sel):
        by = {}
        for k, m, h in R:
            if h == sel:
                by.setdefault(k, []).append(m)
        return [np.mean(by[k]) if k in by else np.nan for k in xs]
    nh = sum(r[2] for r in R); ne = len(R) - nh
    mh = np.mean([r[1] for r in R if r[2]]); me = np.mean([r[1] for r in R if not r[2]])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(xs, curve(False), "-o", color=GREY, ms=5, lw=2, label=f"everything else (n={ne})")
    ax.plot(xs, curve(True), "-o", color=RED, ms=5, lw=2, label=f"hallucination (model, given context) (n={nh})")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel(f"item position ({unit} index)", fontsize=11)
    ax.set_ylabel("mean marginal FVE", fontsize=11)
    ax.set_title(f"{title} — hallucinations vs everything else\n"
                 f"within-position gap {coef:+.4f} (p={p:.2f}); mean {mh:+.3f} vs {me:+.3f}", fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, -0.01,
             "Hallucination = item still judged unfaithful after the judge saw the model's own generated output (§3g). "
             "Everything else = all other items.", fontsize=7.6, color="#777", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_halluc_vs_rest_{arm}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] fig_halluc_vs_rest_{arm}.png  halluc n={nh} mean {mh:+.4f} | else n={ne} mean {me:+.4f} | within-pos {coef:+.4f} p={p:.3f}")
