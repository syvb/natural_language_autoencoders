"""Quick-signal experiment, phase C: does a variation's eval-y score correlate with
WHERE the matryoshka NLA first flags eval-awareness?

For each of the 100 variations we have (a) an eval-y score from the judge panel and
(b) matryoshka-RL verbalizations at ~30 token positions. We judge every explanation
LINE for eval-awareness, then per position find the index of the FIRST eval-aware
line. Aggregating across positions per variation (absence handled explicitly), we
correlate that "where" metric against the eval-y score.

Absence handling: a position whose explanation has NO eval-aware line is censored at
n_lines (the flag is "beyond the list"). We report three metrics:
  - mean_first_idx   : mean first-flag line index, absent positions censored at n_lines
                       (primary; lower = flagged earlier/higher-salience)
  - frac_flag        : fraction of positions with ANY eval-aware line
  - mean_idx_present : mean first-flag index over only the positions that flagged

Hypothesis: more eval-y -> flagged earlier => eval_score correlates NEGATIVELY with
mean_first_idx and POSITIVELY with frac_flag.

    OPENROUTER_API_KEY=$(cat ~/.openrouter_key) python3 realism_variants_correlate.py
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
import eval_awareness as ea  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "realism_variants_line_scores.json"
THR = 0.5


def units_of(lines, mode):
    """Split one position's explanation into scored UNITS.
    - lines:     the NLA's own lines (matryoshka bullets; standard sentences).
    - chunks10:  join the whole explanation and split into 10 ~equal word-chunks,
                 giving the standard NLA a 10-unit 'where' scale like matryoshka's."""
    clean = [l.strip() for l in lines if l and l.strip()]
    if mode == "lines":
        return clean
    words = " ".join(clean).split()
    if not words:
        return [""] * 10
    return [" ".join(g) for g in np.array_split(np.array(words, dtype=object), 10)]


async def get_scores(variants, mode):
    need = set()
    for v in variants:
        for pos in v["lines"]:
            for u in units_of(pos, mode):
                if u.strip():
                    need.add(u.strip())
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    todo = [s for s in need if s not in cache]
    print(f"{len(need)} unique lines, {len(todo)} to judge (~${len(todo)*150/1e6*0.15:.2f})", flush=True)
    if todo:
        key = os.environ["OPENROUTER_API_KEY"]
        sem = asyncio.Semaphore(32); done = [0]
        async with httpx.AsyncClient() as client:
            async def run(s):
                cache[s] = await ea.judge(client, sem, key, s)
                done[0] += 1
                if done[0] % 4000 == 0:
                    print(f"  judged {done[0]}/{len(todo)}", flush=True)
            await asyncio.gather(*(run(s) for s in todo))
        json.dump(cache, open(CACHE, "w"))
        print(f"[cached] {CACHE.name}", flush=True)
    return cache


def variant_metrics(v, cache, mode):
    first_censored, flags, first_present = [], [], []
    for pos in v["lines"]:
        units = units_of(pos, mode)
        n = len(units)
        if n == 0:
            continue
        aware = [i for i, u in enumerate(units)
                 if (cache.get(u.strip()) is not None and cache[u.strip()] >= THR)]
        if aware:
            flags.append(1); first_censored.append(aware[0]); first_present.append(aware[0])
        else:
            flags.append(0); first_censored.append(n)      # censored at n_lines
    return {
        "eval_score": v["eval_score"],
        "mean_first_idx": float(np.mean(first_censored)) if first_censored else float("nan"),
        "frac_flag": float(np.mean(flags)) if flags else float("nan"),
        "mean_idx_present": float(np.mean(first_present)) if first_present else float("nan"),
        "n_pos": len(flags),
    }


def corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return len(x), pr, pp, sr, sp


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mat_rl")
    ap.add_argument("--unit", default="lines", choices=["lines", "chunks10"])
    args = ap.parse_args()
    tag = args.model if args.unit == "lines" else f"{args.model}_{args.unit}"
    variants = json.load(open(HERE / f"realism_variants_verb_{args.model}.json"))["variants"]
    cache = await get_scores(variants, args.unit)
    rows = [variant_metrics(v, cache, args.unit) for v in variants]
    es = [r["eval_score"] for r in rows]

    print(f"\n=== [{tag}] correlation of eval-y score with NLA 'where' metrics (n=%d) ===" % len(rows))
    for metric, lbl in [("mean_first_idx", "mean first-flag line idx (absent=n_lines)"),
                        ("frac_flag", "fraction of positions flagging"),
                        ("mean_idx_present", "mean first-flag idx | flagged")]:
        vals = [r[metric] for r in rows]
        n, pr, pp, sr, sp = corr(es, vals)
        print(f"  {lbl:42s}  Pearson r={pr:+.3f} (p={pp:.1e})  Spearman rho={sr:+.3f} (p={sp:.1e})  n={n}")

    json.dump({"model": args.model, "unit": args.unit, "rows": rows},
              open(HERE / f"realism_variants_correlate_{tag}.json", "w"), indent=1)

    # scatter figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for a, metric, ttl, yl in [
        (ax[0], "mean_first_idx", "eval-y score vs WHERE the NLA first flags it",
         "mean first-flag line index\n(absent censored at n_lines; lower = earlier)"),
        (ax[1], "frac_flag", "eval-y score vs HOW OFTEN the NLA flags it",
         "fraction of positions flagging eval-awareness")]:
        x = np.array(es); y = np.array([r[metric] for r in rows])
        m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
        a.scatter(x, y, s=26, alpha=0.7, color="#2a78d6", edgecolor="white", linewidth=0.4)
        b, a0 = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        a.plot(xs, b * xs + a0, color="#e34948", lw=2)
        n, pr, pp, sr, sp = corr(x, y)
        a.set(xlabel="eval-y score (0 real → 100 eval)", ylabel=yl, title=ttl)
        a.text(0.03, 0.03, f"Pearson r={pr:+.3f} (p={pp:.1e})\nSpearman ρ={sr:+.3f}",
               transform=a.transAxes, fontsize=10, va="bottom",
               bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    fig.suptitle(f"NLA = {args.model} (unit: {args.unit})", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / f"realism_variants_corr_{tag}.png", dpi=140, bbox_inches="tight")
    print(f"[saved] realism_variants_corr_{tag}.png + realism_variants_correlate_{tag}.json")


if __name__ == "__main__":
    asyncio.run(main())
