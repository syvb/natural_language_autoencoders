"""Matryoshka vs standard NLA: what does deleting one explanation unit cost?

Matryoshka explanations are independent, salience-ordered lines (unit = line);
the standard NLA writes `<explanation>` prose (unit = sentence, quote-aware
split — see std_loo_precompute.py). Both models have near-identical full-
explanation own-critic FVE on the same texts (~0.50-0.53), so raw damage is
directly comparable.

For every (position, unit): damage d = FVE(all) − FVE(all∖unit), against the
unit's marginal credit m from the model's own cumulative unit-prefix curve.
A critic that handles subsets in-distribution loses at most what the unit
carried (d ≤ m; matryoshka is sub-additive — redundancy). A critic that goes
OOD on subset inputs loses MORE than the unit carried (d > m), most damningly
on units that carried ~nothing.

    python ablation_compare.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_matryoshka():
    pre = json.load(open(HERE / "space" / "precache.json"))["entries"]
    loo = json.load(open(HERE / "space" / "loo.json"))["entries"]
    d, m, solo, full = [], [], [], []
    for pe, le in zip(pre, loo):
        for r, f, l, s in zip(pe["results"], le["full"], le["loo"], le["solo"]):
            if not r or l is None:
                continue
            fve = r["fve"]
            marg = [fve[0]] + [fve[k] - fve[k - 1] for k in range(1, len(fve))]
            for k in range(len(l)):
                d.append(f - l[k]); m.append(marg[k]); solo.append(s[k]); full.append(f)
    return map(np.array, (d, m, solo, full))


def load_standard():
    loo = json.load(open(HERE / "space_std" / "loo.json"))["entries"]
    d, m, solo, full = [], [], [], []
    for le in loo:
        for f, l, s, p in zip(le["full"], le["loo"], le["solo"], le["pfx"]):
            if f is None or l is None:
                continue
            marg = [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]
            for k in range(len(l)):
                d.append(f - l[k]); m.append(marg[k]); solo.append(s[k]); full.append(f)
    return map(np.array, (d, m, solo, full))


def stats(name, d, m, solo, full):
    lowm = m < 0.01  # units whose prefix credit says they carry ~nothing
    print(f"\n[{name}]  n={len(d)} units, mean full FVE {full.mean():.3f}")
    print(f"  damage d:            mean {d.mean():+.4f}  p50 {np.percentile(d, 50):+.4f}  "
          f"p90 {np.percentile(d, 90):+.4f}")
    print(f"  super-additivity d−m: mean {(d - m).mean():+.4f}  p50 {np.percentile(d - m, 50):+.4f}  "
          f"frac(d > m+0.05) {np.mean(d > m + 0.05):.1%}")
    print(f"  free deletions (d<0.01): {np.mean(d < 0.01):.1%}")
    print(f"  zero-credit units (m<0.01, n={lowm.sum()}): median damage "
          f"{np.percentile(d[lowm], 50):+.4f}, p90 {np.percentile(d[lowm], 90):+.4f}, "
          f"frac damaging >0.05 FVE: {np.mean(d[lowm] > 0.05):.1%}")
    print(f"  damage as frac of full FVE: median {np.percentile(d / np.maximum(full, 1e-6), 50):+.3f}")
    print(f"  solo FVE: median {np.percentile(solo, 50):+.3f}")
    return lowm


def main():
    mat = tuple(load_matryoshka())
    std = tuple(load_standard())
    stats("matryoshka (line units)", *mat)
    stats("standard (sentence units)", *std)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    for ax, (d, m, _, _), name in zip(axes[0], [mat, std],
                                      ["matryoshka — delete one LINE",
                                       "standard — delete one SENTENCE"]):
        lim = (-0.12, 0.55)
        hb = ax.hexbin(m, d, gridsize=55, bins="log", cmap="viridis",
                       extent=(lim[0], lim[1], lim[0], lim[1]))
        fig.colorbar(hb, ax=ax, label="log(count)")
        ax.plot(lim, lim, "r--", lw=1.5, label="damage = credit (lossless deletion)")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(xlabel="marginal credit m (unit-prefix curve)",
               ylabel="deletion damage d = FVE(all) − FVE(all∖unit)",
               title=f"{name}\nabove the line = deletion destroys MORE than the unit carried")
        ax.legend(loc="upper left", fontsize=9)

    ax = axes[1][0]  # damage on zero-credit units
    bins = np.linspace(-0.1, 0.5, 61)
    for (d, m, _, _), name, c in zip([mat, std], ["matryoshka", "standard"],
                                     ["#2b6cb0", "#d64545"]):
        ax.hist(d[m < 0.01], bins=bins, density=True, histtype="step", lw=2,
                color=c, label=f"{name} (n={np.sum(m < 0.01)})")
    ax.set(yscale="log", xlabel="deletion damage d", ylabel="density (log)",
           title="damage from deleting a ZERO-CREDIT unit (m < 0.01)\n"
                 "an in-distribution critic should lose ~nothing")
    ax.legend()

    ax = axes[1][1]  # summary bars
    labels = ["median d", "median d−m", "frac d>m+.05", "frac d<.01\n(free)",
              "median d | m<.01"]
    vals_m = [np.percentile(mat[0], 50), np.percentile(mat[0] - mat[1], 50),
              np.mean(mat[0] > mat[1] + 0.05), np.mean(mat[0] < 0.01),
              np.percentile(mat[0][mat[1] < 0.01], 50)]
    vals_s = [np.percentile(std[0], 50), np.percentile(std[0] - std[1], 50),
              np.mean(std[0] > std[1] + 0.05), np.mean(std[0] < 0.01),
              np.percentile(std[0][std[1] < 0.01], 50)]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, vals_m, 0.36, color="#2b6cb0", label="matryoshka")
    ax.bar(x + 0.18, vals_s, 0.36, color="#d64545", label="standard")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(title="unit-deletion robustness summary", ylabel="FVE / fraction")
    ax.legend()

    fig.tight_layout()
    png = HERE / "ablation_matryoshka_vs_standard.png"
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
