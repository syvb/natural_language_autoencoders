"""Redo the matryoshka 'where' correlation, but only counting lines whose MARGINAL
FVE > threshold (load-bearing lines) — does restricting to substantive lines change
the eval-y-vs-where relationship, and does anything survive excluding blatant prompts?

  OPENROUTER_API_KEY=$(cat ~/.openrouter_key) python3 realism_variants_fve_corr.py --fve-thr 0.05
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
THR = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fve-thr", type=float, default=0.05)
    ap.add_argument("--absence", type=float, default=10.0)
    args = ap.parse_args()

    cache = json.load(open(HERE / "realism_variants_line_scores.json"))
    verb = {v["id"]: v for v in json.load(open(HERE / "realism_variants_verb_mat_rl.json"))["variants"]}
    fve = {v["id"]: v for v in json.load(open(HERE / "realism_variants_fve.json"))["variants"]}

    def rows(fve_gate):
        xs, depth, freq = [], [], []
        n_kept, n_total = 0, 0
        for vid, vv in verb.items():
            marg = fve[vid]["marginal_fve"]
            per_pos, fl = [], []
            for pi, lines in enumerate(vv["lines"]):
                ls = [l.strip() for l in lines if l and l.strip()]
                mk = marg[pi] if pi < len(marg) else []
                # keep lines that are load-bearing (marginal FVE > gate); keep original salience index
                aware = []
                for k, ln in enumerate(ls):
                    load_bearing = (not fve_gate) or (k < len(mk) and mk[k] > args.fve_thr)
                    n_total += 1; n_kept += int(load_bearing)
                    if load_bearing and cache.get(ln) is not None and cache[ln] >= THR:
                        aware.append(k)
                if not ls:
                    continue
                per_pos.append(aware[0] if aware else args.absence)
                fl.append(1 if aware else 0)
            if per_pos:
                xs.append(vv["eval_score"]); depth.append(np.mean(per_pos)); freq.append(np.mean(fl))
        return np.array(xs), np.array(depth), np.array(freq), n_kept, n_total

    def report(label, x, y, sub=None):
        m = np.isfinite(x) & np.isfinite(y)
        if sub is not None:
            m &= sub
        r, p = pearsonr(x[m], y[m]); s, sp = spearmanr(x[m], y[m])
        print(f"  {label:34s} n={m.sum():3d}  Pearson r={r:+.3f} (p={p:.2e})  Spearman rho={s:+.3f}")

    for gate, lbl in [(False, "ALL lines (no FVE gate)"), (True, f"marginal FVE > {args.fve_thr}")]:
        x, d, f, nk, nt = rows(gate)
        keep = f"  [{nk}/{nt} lines kept, {100*nk/nt:.0f}%]" if gate else ""
        print(f"\n=== {lbl} ==={keep}")
        report("where (all variations)", x, d)
        report("where (evaliness<=85)", x, d, x <= 85)
        report("frequency (all)", x, f)
        report("frequency (evaliness<=85)", x, f, x <= 85)

    # scatter for the gated version
    x, d, f, nk, nt = rows(True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    lo, hi = x <= 85, x > 85
    ax.scatter(x[lo], d[lo], s=30, alpha=0.8, color="#2a78d6", edgecolor="white", lw=0.4,
               label=f"evaliness ≤ 85 (n={lo.sum()})")
    ax.scatter(x[hi], d[hi], s=34, alpha=0.85, color="#e34948", edgecolor="white", lw=0.4,
               label=f"evaliness > 85 (n={hi.sum()})")
    b, a0 = np.polyfit(x[lo], d[lo], 1); xr = np.array([x[lo].min(), x[lo].max()])
    ax.plot(xr, b * xr + a0, color="#2a78d6", lw=2, ls="--")
    b2, a2 = np.polyfit(x, d, 1); xr2 = np.array([x.min(), x.max()])
    ax.plot(xr2, b2 * xr2 + a2, color="#888", lw=1.6, ls=":")
    r_all, p_all = pearsonr(x, d); r_lo, p_lo = pearsonr(x[lo], d[lo])
    ax.set_xlabel("evaliness score  (0 = real → 100 = obvious eval)")
    ax.set_ylabel(f"first eval-aware line idx among lines w/ marginal FVE>{args.fve_thr}\n(absence={args.absence:.0f})")
    ax.set_title(f"matryoshka RL: 'where' using only load-bearing lines (marginal FVE>{args.fve_thr})")
    ax.text(0.03, 0.03, f"all (n=100):  r={r_all:+.3f} (p={p_all:.1e})\n≤85 (n={lo.sum()}):  r={r_lo:+.3f} (p={p_lo:.2f})",
            transform=ax.transAxes, fontsize=10, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "realism_variants_fve_corr.png", dpi=140)
    print("\n[saved] realism_variants_fve_corr.png")


if __name__ == "__main__":
    main()
