"""Two standalone figures (one per model): mean marginal FVE by position,
comparing SUPPORTED (faithful) vs hallucinated-excluding-misquotes (SUBSTANTIVE).
Same style as the right panels of fig_marginal_by_halluc.png."""
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GREEN, ORANGE = "#2f9c69", "#b4520a"
H = {"CONTRADICTED", "FABRICATED"}
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
reason = json.load(open(HERE / "results" / "quote_reason.json"))


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
                    out.append((k, m[k], v, reason.get(key)))
    return out


def ols(sub_rows, sup_rows):
    rows = [(r[0], r[1], 1) for r in sub_rows] + [(r[0], r[1], 0) for r in sup_rows]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    h = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [h])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b; dof = len(y) - X.shape[1]
    cov = (resid @ resid) / dof * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[-1, -1]))
    return float(b[-1]), float(2 * stats.t.sf(abs(b[-1] / se), dof))


SPECS = [("mat", "matryoshka (lines)", "subset_scores_mat.json", "line"),
         ("std", "standard (sentences)", "subset_scores_std.json", "sentence")]

for arm, name, fn, unit in SPECS:
    R = load(arm, fn)
    sup = [r for r in R if r[2] == "SUPPORTED"]
    subst = [r for r in R if r[2] in H and r[3] == "SUBSTANTIVE"]
    ks = np.array([r[0] for r in R])
    kmax = int(np.percentile(ks, 99))
    xs = list(range(kmax + 1))

    def curve(rows):
        by = {}
        for k, m, *_ in rows:
            by.setdefault(k, []).append(m)
        return [np.mean(by[k]) if k in by else np.nan for k in xs]

    def counts(rows):
        by = {}
        for k, *_ in rows:
            by[k] = by.get(k, 0) + 1
        return [by.get(k, 0) for k in xs]

    coef, p = ols(subst, sup)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(xs, curve(sup), "-o", color=GREEN, ms=5, lw=2.0, label=f"faithful — SUPPORTED (n={len(sup)})")
    ax.plot(xs, curve(subst), "-o", color=ORANGE, ms=5, lw=2.0,
            label=f"hallucinated, excl. misquotes (n={len(subst)})")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel(f"item position ({unit} index)", fontsize=11)
    ax.set_ylabel("mean marginal FVE", fontsize=11)
    ax.set_title(f"{name} — marginal FVE by position\n"
                 f"faithful vs substantive hallucinations (misquotes excluded)",
                 fontsize=12)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    sig = "significant" if p < 0.05 else "n.s."
    fig.text(0.01, -0.02,
             f"Substantive = hallucination is a fact fabrication/contradiction in the note's own words, NOT an invented quote "
             f"(nex-n2-mini).\nWithin-position gap (OLS marginal ~ C(position) + is_halluc, vs SUPPORTED): "
             f"coef {coef:+.4f}, p={p:.3f} ({sig}). n per point drops with position; "
             f"curves shown to the 99th-percentile position.",
             fontsize=7.8, color="#777", ha="left", va="top")
    fig.tight_layout()
    out = HERE / "results" / f"fig_substantive_by_pos_{arm}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out.name}  (coef {coef:+.4f}, p={p:.3f}, SUPPORTED n={len(sup)}, SUBSTANTIVE n={len(subst)})")
