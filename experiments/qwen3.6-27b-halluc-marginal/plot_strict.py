"""Even the strictest, cross-model-confirmed hallucination set shows no
within-position FVE signal. Forest plot of the within-position coef (vs
SUPPORTED) as the hallucinated bucket is progressively purified, with the
permutation null band (random same-size subset of the loose bucket)."""
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
strict = json.load(open(HERE / "results" / "strict_halluc.json"))
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))


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
    rows = [(p, m, 0) for p, m in sup] + [(it["pos"], it["mg"], 1) for it in grp]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), se


fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
rng = random.Random(0)
for ax, (arm, fn, title) in zip(axes, [("mat", "subset_scores_mat.json", "matryoshka (lines)"),
                                       ("std", "subset_scores_std.json", "standard (sentences)")]):
    R = load(arm, fn)
    sup = [(it["pos"], it["mg"]) for it in R if it["v"] == "SUPPORTED"]
    loose = [it for it in R if it["v"] in H]
    groups = [
        (f"loose: all CON/FAB (n={len(loose)})", loose),
        (f"nex strict (n={sum(strict[it['key']][MODELS[0]] for it in loose)})",
         [it for it in loose if strict[it["key"]][MODELS[0]]]),
        (f"gpt-4o-mini strict (n={sum(strict[it['key']][MODELS[1]] for it in loose)})",
         [it for it in loose if strict[it["key"]][MODELS[1]]]),
        (f"CONFIRMED: both agree (n={sum(all(strict[it['key']].values()) for it in loose)})",
         [it for it in loose if all(strict[it["key"]].values())]),
    ]
    # permutation null band from a random same-size subset (size of CONFIRMED)
    nconf = len(groups[-1][1])
    perm = []
    for _ in range(600):
        idx = list(range(len(loose))); rng.shuffle(idx)
        perm.append(ols(sup, [loose[i] for i in idx[:nconf]])[0])
    lo, hi = np.percentile(perm, [2.5, 97.5])
    ax.axvspan(lo, hi, color="#bbb", alpha=0.30, label="random-subset null (95%)")
    ys = list(range(len(groups)))[::-1]
    cols = ["#e34948", "#e39a48", "#8a6d3b", "#2a78d6"]
    for y, (name, grp), col in zip(ys, groups, cols):
        b, se = ols(sup, grp)
        p = 2 * stats.t.sf(abs(b / se), len(sup) + len(grp))
        ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color=col, ms=8, capsize=4, lw=2)
        sig = "*" if p < 0.05 else ""
        ax.text(b, y + 0.2, f"{b:+.4f}{sig} (p={p:.2f})", ha="center", fontsize=8.5, color=col)
    ax.axvline(0, color="#444", lw=1.1, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([g[0] for g in groups], fontsize=9)
    ax.set_xlabel("within-position FVE, hallucinated − SUPPORTED", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", color="#ccc", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.18)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Purifying the hallucination bucket doesn't reveal a signal — even the cross-model-confirmed\n"
             "set is indistinguishable from a random subset (FVE tracks reconstructability, not faithfulness)",
             fontsize=12, y=1.04)
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_strict_consensus.png", dpi=150, bbox_inches="tight")
print("[saved] results/fig_strict_consensus.png")
