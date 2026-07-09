"""Per-LINE eval-awareness vs marginal-FVE (matryoshka).

For every line of every NLA explanation in a precache, judge P(eval-awareness)
(gpt-4o-mini logprob, generic prompt — reused from eval_awareness.py) and pair it
with that line's MARGINAL FVE (fve[k]-fve[k-1], the app's "marginal" view). Then
show the relationship: hexbin + binned trend + P(aware) by salience rank, with
Pearson/Spearman r. Saves the paired data + a PNG.

    OPENROUTER_API_KEY=... python eval_awareness_lines.py space/precache.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_awareness as ea  # judge(), SYS/USR/YES/NO, URL/MODEL  # noqa: E402


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("precache")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    entries = json.load(open(args.precache))["entries"]

    # collect (line, marginal_fve, line_idx) for every line; dedup lines to judge
    items = []
    for e in entries:
        for r in e["results"]:
            if not r:
                continue
            prev = 0.0
            for k, (ln, f) in enumerate(zip(r["lines"], r["fve"])):
                items.append((ln, f - prev, k))
                prev = f
    uniq = {ln: None for ln, _, _ in items}
    keys = list(uniq)
    print(f"[{Path(args.precache).parent.name}] {len(items)} lines, {len(keys)} unique → judging", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    done = [0]
    async with httpx.AsyncClient() as client:
        async def run(s):
            uniq[s] = await ea.judge(client, sem, key, s)
            done[0] += 1
            if done[0] % 2000 == 0:
                print(f"  judged {done[0]}/{len(keys)}", flush=True)
        await asyncio.gather(*(run(s) for s in keys))

    mfve = np.array([m for ln, m, k in items if uniq[ln] is not None], dtype=float)
    paw = np.array([uniq[ln] for ln, m, k in items if uniq[ln] is not None], dtype=float)
    lidx = np.array([k for ln, m, k in items if uniq[ln] is not None], dtype=int)
    print(f"[data] {len(paw)} scored line-instances (fails={sum(v is None for v in uniq.values())} uniques)", flush=True)

    # save paired data
    out = args.out or str(Path(args.precache).with_name("eval_awareness_lines.json"))
    json.dump({"model": ea.MODEL, "marginal_fve": mfve.round(5).tolist(),
               "p_aware": paw.round(5).tolist(), "line_idx": lidx.tolist()}, open(out, "w"))

    # stats
    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(mfve, paw); sr = spearmanr(mfve, paw)
    aw = paw >= 0.5
    print(f"\n=== relationship (n={len(paw)}) ===")
    print(f"  Pearson r  (marginal FVE, P_aware) = {pr.statistic:+.3f}  (p={pr.pvalue:.1e})")
    print(f"  Spearman r                         = {sr.statistic:+.3f}  (p={sr.pvalue:.1e})")
    print(f"  mean marginal FVE | eval-aware (P>=.5): {mfve[aw].mean():+.4f}  (n={aw.sum()})")
    print(f"  mean marginal FVE | not          (P<.5): {mfve[~aw].mean():+.4f}  (n={(~aw).sum()})")
    print("  mean P_aware by salience rank (line idx):")
    for k in range(int(lidx.max()) + 1):
        msk = lidx == k
        if msk.sum():
            print(f"    line {k}: n={msk.sum():5d}  meanP={paw[msk].mean():.3f}  meanMargFVE={mfve[msk].mean():+.4f}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    lo, hi = np.percentile(mfve, [1, 99])
    hb = ax1.hexbin(mfve, paw, gridsize=45, bins="log", cmap="viridis",
                    extent=(lo, hi, 0, 1))
    fig.colorbar(hb, ax=ax1, label="log(count)")
    # binned mean trend
    bins = np.linspace(lo, hi, 16)
    idx = np.digitize(mfve, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 20:
            bx.append((bins[b - 1] + bins[b]) / 2); by.append(paw[m].mean())
    ax1.plot(bx, by, "o-", color="orange", lw=2, ms=5, label="binned mean P")
    ax1.set(xlabel="marginal FVE (fve[k]−fve[k−1])", ylabel="P(eval-aware)",
            title=f"per-line: marginal FVE vs eval-awareness\nPearson r={pr.statistic:+.3f}, Spearman r={sr.statistic:+.3f}  (n={len(paw)})")
    ax1.legend(loc="upper right")
    # P_aware + marginal FVE by salience rank
    ks = list(range(int(lidx.max()) + 1))
    mp = [paw[lidx == k].mean() for k in ks]
    mm = [mfve[lidx == k].mean() for k in ks]
    ax2.bar(ks, mp, color="#d64545", alpha=.8, label="mean P(eval-aware)")
    ax2.set(xlabel="line index (salience rank; 0 = most salient)", ylabel="mean P(eval-aware)",
            title="eval-awareness & marginal FVE by salience rank")
    ax2b = ax2.twinx()
    ax2b.plot(ks, mm, "s-", color="#2b6cb0", label="mean marginal FVE")
    ax2b.set_ylabel("mean marginal FVE", color="#2b6cb0")
    ax2.legend(loc="upper left"); ax2b.legend(loc="upper right")
    fig.tight_layout()
    png = args.png or str(Path(args.precache).with_name("eval_awareness_vs_fve.png"))
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {out}  and  {png}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
