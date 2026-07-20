"""What does marginal FVE actually track? Scan candidate item categories and rank
by within-position effect size (OLS marginal ~ C(position) + feature). Surface /
reconstructability features (quoted string present, item length, verbatim copying
from the source) dwarf every faithfulness-based category — the critic's marginal
FVE indexes how reconstructable the TEXT is, not whether it is faithful."""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SE = HERE.parent / "qwen3.6-27b-suffix-eval"
man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
reason = json.load(open(HERE / "results" / "quote_reason.json"))
sev = json.load(open(HERE / "results" / "severity.json"))
H = {"CONTRADICTED", "FABRICATED"}
QUOTE = re.compile(r'["“”‘’«»]')
SURFACE, CONTENT, FAITH = "#2a78d6", "#8a6d3b", "#e34948"


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def load(arm, fn):
    out = []
    for e in json.load(open(HERE / "results" / fn))["entries"]:
        sw = man[e["ci"]]["prefix_text"].lower().split()
        grams = {tuple(sw[i:i + 4]) for i in range(len(sw) - 3)}
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            m = marg(rec)
            for k in range(len(m)):
                key = f"{arm}|{e['ci']}|{ri}|{k}"
                v = faith.get(key)
                if v is None:
                    continue
                t = rec["units"][k]
                tw = t.lower().split()
                tg = {tuple(tw[i:i + 4]) for i in range(len(tw) - 3)}
                out.append(dict(pos=k, mg=m[k], v=v, sev=sev.get(key),
                                quote=bool(QUOTE.search(t)), digit=bool(re.search(r"\d", t)),
                                nchar=len(t), copy4=len(tg & grams) > 0))
    return out


def ols(rows, featfn):
    sub = [(r["pos"], r["mg"], featfn(r)) for r in rows if featfn(r) is not None]
    ks = np.array([r[0] for r in sub]); y = np.array([r[1] for r in sub]); f = np.array([r[2] for r in sub], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), se, float(2 * stats.t.sf(abs(b[-1] / se), dof)), int(f.sum())


FEATS = [
    ("contains quoted string", "quote", SURFACE, lambda r: r["quote"]),
    ("copies 4+ words from source", "copy", SURFACE, lambda r: r["copy4"]),
    ("long item (> median chars)", "long", SURFACE, None),  # threshold filled per-model
    ("contains a digit", "digit", SURFACE, lambda r: r["digit"]),
    ("META (genre/tone commentary)", "meta", CONTENT, lambda r: r["v"] == "META"),
    ("hallucinated (CON/FAB)", "halluc", FAITH, lambda r: r["v"] in H),
    ("genuine confab vs prediction", "genuine", FAITH,
     lambda r: (r["sev"] == "GENUINE_FABRICATION") if r["sev"] else None),
]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
for ax, (arm, fn, title) in zip(axes, [("mat", "subset_scores_mat.json", "matryoshka (lines)"),
                                        ("std", "subset_scores_std.json", "standard (sentences)")]):
    R = load(arm, fn)
    med = np.median([r["nchar"] for r in R])
    print(f"==== {title}  (n={len(R)}, median chars={med:.0f}) ====")
    rows = []
    for name, _, col, fnc in FEATS:
        f = (lambda r, m=med: r["nchar"] > m) if fnc is None else fnc
        b, se, p, n1 = ols(R, f)
        rows.append((name, col, b, se, p, n1))
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        print(f"  {name:32s} coef {b:+.4f}  p={p:.1e} {star}")
    rows.sort(key=lambda x: x[2])
    ys = range(len(rows))
    for y, (name, col, b, se, p, n1) in zip(ys, rows):
        ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color=col, ms=7, capsize=3, lw=2)
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
        ax.text(b, y + 0.24, f"{b:+.3f}{sig}", ha="center", fontsize=8, color=col)
    ax.axvline(0, color="#444", lw=1.1, ls="--")
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("within-position FVE effect (OLS coef, feature − rest)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", color="#ccc", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.12)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0], [0], marker="o", color=SURFACE, ls="", label="surface / reconstructability"),
                        Line2D([0], [0], marker="o", color=CONTENT, ls="", label="content type"),
                        Line2D([0], [0], marker="o", color=FAITH, ls="", label="faithfulness")],
               fontsize=8.5, loc="lower right")
fig.suptitle("Marginal FVE tracks how reconstructable the TEXT is, not whether it's faithful\n"
             "surface features (quotes, length, verbatim copying) dwarf every faithfulness category",
             fontsize=12.5, y=1.03)
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_what_marginal_tracks.png", dpi=150, bbox_inches="tight")
print("\n[saved] results/fig_what_marginal_tracks.png")
