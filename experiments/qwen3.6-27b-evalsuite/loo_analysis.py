"""LOO necessity vs prefix-marginal ΔFVE — the redundancy structure of
matryoshka explanations (the figure that motivates the ablation view).

x = marginal ΔFVE (prefix-based credit, what the app's marginal view shows)
y = LOO necessity (full − FVE-without-this-line, what ablation measures)

Points on the diagonal: the line's prefix credit is exactly what you lose by
deleting it. Below: REDUNDANT (later lines cover it — prefix credit
overstates necessity). Above: SYNERGISTIC (the line is worth more in context
than its prefix credit). Also reports solo-FVE (sufficiency) by salience rank
and the count of "deletable" top lines (rank-0 lines with necessity < 0.01).

    python loo_analysis.py space
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("space", nargs="?", default="space")
    ap.add_argument("--max-rank", type=int, default=None,
                    help="keep only lines with salience rank < this (e.g. 5)")
    args = ap.parse_args()
    space = Path(args.space)
    pre = json.load(open(space / "precache.json"))["entries"]
    loo = json.load(open(space / "loo.json"))["entries"]

    marg, nec, solo, rank = [], [], [], []
    for pe, le in zip(pre, loo):
        for r, f_full, l_arr, s_arr in zip(pe["results"], le["full"], le["loo"], le["solo"]):
            if not r or l_arr is None:
                continue
            fve = r["fve"]
            m = [fve[0]] + [fve[k] - fve[k - 1] for k in range(1, len(fve))]
            for k in range(len(l_arr)):
                marg.append(m[k]); nec.append(f_full - l_arr[k])
                solo.append(s_arr[k]); rank.append(k)
    marg, nec, solo, rank = map(np.array, (marg, nec, solo, rank))
    if args.max_rank is not None:
        keep = rank < args.max_rank
        marg, nec, solo, rank = marg[keep], nec[keep], solo[keep], rank[keep]
    print(f"[data] {len(marg)} (position,line) pairs"
          + (f" (ranks 0..{args.max_rank - 1})" if args.max_rank else ""))

    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(marg, nec); sr = spearmanr(marg, nec)
    print(f"marginal vs necessity: Pearson r={pr.statistic:+.3f} Spearman r={sr.statistic:+.3f}")
    red = marg - nec  # >0 ⇒ prefix credit exceeds causal necessity (redundant)
    print(f"redundancy (marginal − necessity): mean={red.mean():+.4f} "
          f"p50={np.percentile(red, 50):+.4f} p90={np.percentile(red, 90):+.4f}")
    print("\nby salience rank:  meanMarg  meanNecessity  meanSolo  frac(necessity<0.01)")
    for k in range(int(rank.max()) + 1):
        msk = rank == k
        print(f"  line {k}: {marg[msk].mean():+.4f}   {nec[msk].mean():+.4f}      "
              f"{solo[msk].mean():+.4f}   {(nec[msk] < 0.01).mean():.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    lim = np.percentile(np.concatenate([marg, nec]), [0.5, 99.5])
    hb = ax1.hexbin(marg, nec, gridsize=55, bins="log", cmap="viridis",
                    extent=(lim[0], lim[1], lim[0], lim[1]))
    fig.colorbar(hb, ax=ax1, label="log(count)")
    ax1.plot(lim, lim, "r--", lw=1.5, label="necessity = marginal")
    ax1.set(xlabel="marginal ΔFVE (prefix credit)",
            ylabel="LOO necessity (full − FVE w/o line)",
            title=f"per-line: prefix credit vs causal necessity\n"
                  f"Pearson r={pr.statistic:+.3f}  (n={len(marg)}); "
                  f"below the line = redundant")
    ax1.legend(loc="upper left")

    ks = list(range(int(rank.max()) + 1))
    w = 0.27
    ax2.bar([k - w for k in ks], [marg[rank == k].mean() for k in ks], w,
            color="#2b6cb0", label="marginal ΔFVE (prefix credit)")
    ax2.bar(ks, [nec[rank == k].mean() for k in ks], w,
            color="#d64545", label="LOO necessity (delete → lose)")
    ax2.bar([k + w for k in ks], [solo[rank == k].mean() for k in ks], w,
            color="#7aa25c", label="solo FVE (line alone)")
    ax2.set(xlabel="line index (salience rank; 0 = most salient)", ylabel="mean FVE",
            title="credit vs necessity vs sufficiency by salience rank")
    ax2.legend()
    fig.tight_layout()
    suffix = f"_r{args.max_rank}" if args.max_rank is not None else ""
    png = space.parent / f"loo_vs_marginal{suffix}.png"
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
