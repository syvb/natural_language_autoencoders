"""Forest plot: within-position marginal-FVE gap (vs SUPPORTED) for hallucination
subtypes — ALL / SUBSTANTIVE (excl. unfaithful quotes) / QUOTE-driven, both models.
Also reports the cruder quote-mark-presence split for contrast."""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED, GREEN, GREY = "#2a78d6", "#e34948", "#2f9c69", "#8896a3"
H = {"CONTRADICTED", "FABRICATED"}
QUOTE = re.compile(r'["“”‘’«»]')
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
                v = faith.get(f"{arm}|{e['ci']}|{ri}|{k}")
                if v is not None:
                    out.append(dict(pos=k, mg=m[k], v=v, text=rec["units"][k],
                                    key=f"{arm}|{e['ci']}|{ri}|{k}"))
    return out


def ols_ci(target, sup):
    """coef, se, p, n for is_target in marginal ~ C(pos)+is_target."""
    rows = [(r["pos"], r["mg"], 1) for r in target] + [(r["pos"], r["mg"], 0) for r in sup]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    h = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [h])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b; dof = len(y) - X.shape[1]
    cov = (resid @ resid) / dof * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[-1, -1]))
    p = float(2 * stats.t.sf(abs(b[-1] / se), dof))
    return float(b[-1]), se, p, len(target)


rowspecs = []  # (label, coef, se, p, n, color)
data = {"matryoshka (lines)": ("mat", "subset_scores_mat.json", BLUE),
        "standard (sentences)": ("std", "subset_scores_std.json", RED)}
print("within-position OLS coef (marginal vs SUPPORTED, same position):\n")
for name, (arm, fn, col) in data.items():
    rows = load(arm, fn)
    sup = [r for r in rows if r["v"] == "SUPPORTED"]
    hall = [r for r in rows if r["v"] in H]
    subst = [r for r in hall if reason.get(r["key"]) == "SUBSTANTIVE"]
    quote = [r for r in hall if reason.get(r["key"]) == "QUOTE"]
    noqmark = [r for r in hall if not QUOTE.search(r["text"])]
    print(f"== {name} ==  (SUPPORTED n={len(sup)})")
    for lab, grp in [("all hallucinated", hall), ("substantive (excl. misquotes)", subst),
                     ("quote-driven (misquotes)", quote),
                     ("[proxy] no quote-mark at all", noqmark)]:
        c, se, p, n = ols_ci(grp, sup)
        print(f"  {lab:34s} n={n:4d}  coef {c:+.4f} ± {1.96*se:.4f}  p={p:.3f}")
        if not lab.startswith("[proxy]"):
            rowspecs.append((f"{name.split()[0]}: {lab}", c, se, p, n,
                             GREEN if "substantive" in lab else GREY if "quote" in lab else col))

# ── forest plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 5.2))
ys = list(range(len(rowspecs)))[::-1]
for y, (lab, c, se, p, n, col) in zip(ys, rowspecs):
    ax.errorbar(c, y, xerr=1.96 * se, fmt="o", color=col, ms=7, capsize=4, lw=2)
    sig = "*" if p < 0.05 else ""
    ax.text(c, y + 0.22, f"{c:+.3f}{sig} (p={p:.2f}, n={n})", ha="center", fontsize=8, color=col)
ax.axvline(0, color="#444", lw=1.2, ls="--")
ax.set_yticks(ys)
ax.set_yticklabels([lab for lab, *_ in rowspecs], fontsize=9.5)
ax.set_xlabel("within-position marginal FVE, hallucination minus SUPPORTED  (◀ lower = reconstructs worse)", fontsize=10)
ax.set_title("Excluding unfaithful QUOTES from “hallucination”: a weak signal appears for substantive fabrications\n"
             "(marginal ~ C(position) + is_halluc; * = p<0.05)", fontsize=11)
ax.grid(axis="x", color="#ccc", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.15)
fig.text(0.5, -0.04,
         "Substantive = the note fabricates/contradicts facts in its own words; QUOTE = unfaithfulness is an invented/misattributed "
         "quotation (nex-n2-mini). Excluding misquotes,\nsubstantive hallucinations reconstruct slightly WORSE than faithful "
         "content at the same position — significant for matryoshka (p=0.02), borderline for standard (p=0.06) — while misquotes "
         "reconstruct as well or better.\nThe cruder “no quote-mark at all” proxy exaggerates this (std −0.37): a quoted STRING is "
         "specific, reconstructable text regardless of truth, so quote-absence tracks abstractness, not fabrication.",
         fontsize=7.6, color="#777", ha="center", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_quote_halluc.png", dpi=150, bbox_inches="tight")
print("\n[saved] results/fig_quote_halluc.png")
