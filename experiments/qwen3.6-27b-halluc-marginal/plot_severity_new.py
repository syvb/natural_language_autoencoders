"""fig_severity_{mat,std} rebuilt with the NEW (model-aware) data only. The
unverbalized-prediction vs genuine-confabulation split is no longer a severity
judge's guess — it is empirical: items originally flagged hallucinated that FLIP
to SUPPORTED once the judge sees the model's own generated output are the
confirmed unverbalized predictions; the residual (still hallucinated given the
model's output) are the genuine fabrications."""
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GREEN, ORANGE, DARK = "#2f9c69", "#d98a3a", "#8a1c13"
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
                if v is not None:
                    out.append(dict(key=key, pos=k, mg=m[k], v=v))
    return out


def ols(sup, grp):
    rows = [(r["pos"], r["mg"], 0) for r in sup] + [(r["pos"], r["mg"], 1) for r in grp]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), float(2 * stats.t.sf(abs(b[-1] / se), dof))


for arm, label, fn, unit in [("mat", "matryoshka (lines)", "subset_scores_mat.json", "line"),
                             ("std", "standard (sentences)", "subset_scores_std.json", "sentence")]:
    R = load(arm, fn)
    sup = [r for r in R if r["v"] == "SUPPORTED"]
    loose = [r for r in R if r["v"] in H]
    pred = [r for r in loose if withgen.get(r["key"]) == "SUPPORTED"]      # flipped => confirmed prediction
    genuine = [r for r in loose if withgen.get(r["key"]) in H]             # residual => genuine fabrication
    cg, pg = ols(sup, genuine)
    ks = np.array([r["pos"] for r in R]); kmax = int(np.percentile(ks, 99))
    xs = list(range(kmax + 1))

    def curve(rows):
        by = {}
        for r in rows:
            by.setdefault(r["pos"], []).append(r["mg"])
        return [np.mean(by[k]) if k in by else np.nan for k in xs]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(xs, curve(sup), "-o", color=GREEN, ms=5, lw=2, label=f"faithful — SUPPORTED (n={len(sup)})")
    ax.plot(xs, curve(pred), "-o", color=ORANGE, ms=5, lw=2, label=f"unverbalized prediction — MODEL-CONFIRMED (n={len(pred)})")
    ax.plot(xs, curve(genuine), "-o", color=DARK, ms=5, lw=2, label=f"GENUINE fabrication (n={len(genuine)})")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel(f"item position ({unit} index)", fontsize=11)
    ax.set_ylabel("mean marginal FVE", fontsize=11)
    ax.set_title(f"{label} — worst hallucinations vs model-confirmed predictions (NEW data)\n"
                 f"GENUINE-vs-SUPPORTED within position: coef {cg:+.4f}, p={pg:.2f}", fontsize=11.5)
    ax.legend(fontsize=8.6); ax.grid(color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    import textwrap
    foot = ("Prediction = originally flagged hallucinated but re-judged SUPPORTED once the judge saw the model's own "
            "generated output (§3g) — empirically the model's unverbalized direction. Genuine = still hallucinated given "
            "that output. Note the caveat: items flip partly because they are reconstructable, so the genuine-vs-SUPPORTED "
            "gap is somewhat inflated by the split.")
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.02, 0.02, "\n".join(textwrap.wrap(foot, width=118)), fontsize=7.5, color="#777", ha="left", va="bottom")
    fig.savefig(HERE / "results" / f"fig_severity_{arm}.png", dpi=150)
    plt.close(fig)
    print(f"[saved] fig_severity_{arm}.png  SUPPORTED={len(sup)} pred={len(pred)} genuine={len(genuine)} "
          f"| genuine mean {np.mean([r['mg'] for r in genuine]):+.3f} pred mean {np.mean([r['mg'] for r in pred]):+.3f} "
          f"sup mean {np.mean([r['mg'] for r in sup]):+.3f} | within-pos coef {cg:+.4f} p={pg:.3f}")
