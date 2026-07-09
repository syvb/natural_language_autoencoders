"""Is matryoshka better at DETECTING eval-awareness, or just biased toward it?

Signal-detection comparison of the two spaces' per-token P(eval-aware) scores
(same 5 texts, same judge): plain-control tokens = negatives, honeypot tokens =
positives. Bias shifts both rates together; discrimination separates them.

  - ROC AUC (Mann-Whitney): threshold-free discrimination
  - equal-FPR sensitivity: pick each model's threshold at matched plain-text FPR,
    compare honeypot hit rates
  - per-position honeypot scatter (same tokens, both models)
  - top plain-text false positives w/ the explanation lines that triggered them

    python eval_awareness_bias.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HONEY = "<|im_start|>system"


def load(space):
    ea = json.load(open(HERE / space / "eval_awareness.json"))["entries"]
    plain, honey = [], None
    for e in ea:
        p = np.array([np.nan if v is None else v for v in e["paware"]], dtype=float)
        if e["text"].startswith(HONEY):
            honey = p
        else:
            plain.append(p)
    return np.concatenate(plain), honey


def auc(neg, pos):
    """Mann-Whitney AUC = P(random pos > random neg), ties count half."""
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    order = np.concatenate([neg, pos])
    from scipy.stats import rankdata
    r = rankdata(order)
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def main():
    data = {name: load(space) for name, space in
            [("matryoshka", "space"), ("standard", "space_std")]}

    print("=== base rates & discrimination ===")
    for name, (plain, honey) in data.items():
        pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
        print(f"\n[{name}]  plain n={len(pl)}  honeypot n={len(hp)}")
        print(f"  mean P:            plain {pl.mean():.3f}   honeypot {hp.mean():.3f}")
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            fpr, tpr = (pl >= t).mean(), (hp >= t).mean()
            lift = tpr / fpr if fpr > 0 else float("inf")
            print(f"  P>={t}:  plain {fpr:6.1%}   honeypot {tpr:6.1%}   lift {lift:5.1f}x")
        print(f"  ROC AUC (honeypot vs plain) = {auc(pl, hp):.3f}")

    print("\n=== equal-FPR sensitivity (matched plain-text false-positive rate) ===")
    for target_fpr in (0.05, 0.02, 0.01):
        row = []
        for name, (plain, honey) in data.items():
            pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
            thr = np.quantile(pl, 1 - target_fpr)
            row.append(f"{name}: thr={thr:.3f} hit={100*(hp >= thr).mean():.1f}%")
        print(f"  FPR={target_fpr:.0%}:  " + "   ".join(row))

    # per-position honeypot agreement (same base tokens)
    hm, hs = data["matryoshka"][1], data["standard"][1]
    n = min(len(hm), len(hs))
    m, s = hm[:n], hs[:n]
    ok = ~np.isnan(m) & ~np.isnan(s)
    from scipy.stats import pearsonr, spearmanr
    print(f"\n=== honeypot per-position agreement (n={ok.sum()}) ===")
    print(f"  Pearson r={pearsonr(m[ok], s[ok]).statistic:+.3f}  "
          f"Spearman r={spearmanr(m[ok], s[ok]).statistic:+.3f}")
    both = ((m >= .5) & (s >= .5))[ok].sum()
    monly = ((m >= .5) & (s < .5))[ok].sum()
    sonly = ((m < .5) & (s >= .5))[ok].sum()
    print(f"  flagged (P>=.5): both {both}  matryoshka-only {monly}  standard-only {sonly}")

    # what triggers matryoshka's plain-text false positives?
    print("\n=== matryoshka plain-text false positives (P>=0.5) ===")
    pre = json.load(open(HERE / "space" / "precache.json"))["entries"]
    ea = json.load(open(HERE / "space" / "eval_awareness.json"))["entries"]
    shown = 0
    for e, ee in zip(pre, ea):
        if e["text"].startswith(HONEY):
            continue
        for pos, (r, p) in enumerate(zip(e["results"], ee["paware"])):
            if r and p is not None and p >= 0.5 and shown < 8:
                tok = e["pieces"][pos].strip()
                print(f"\n  [{e['text'][:40]!r} pos {pos} tok {tok!r}]  P={p:.2f}")
                for ln in r["lines"]:
                    print(f"    | {ln}")
                shown += 1
    print(f"\n  (total plain FP count shown above; see rates table for full counts)")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]  # score distributions
    bins = np.linspace(0, 1, 21)
    for i, (name, (plain, honey)) in enumerate(data.items()):
        pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
        c = ["#d64545", "#2b6cb0"][i]
        ax.hist(pl, bins=bins, density=True, histtype="step", lw=2, ls="--",
                color=c, label=f"{name} plain")
        ax.hist(hp, bins=bins, density=True, histtype="step", lw=2,
                color=c, label=f"{name} honeypot")
    ax.set(yscale="log", xlabel="P(eval-aware)", ylabel="density (log)",
           title="score distributions\n(dashed = plain controls, solid = honeypot)")
    ax.legend(fontsize=8)

    ax = axes[1]  # ROC
    for i, (name, (plain, honey)) in enumerate(data.items()):
        pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
        thr = np.unique(np.concatenate([pl, hp]))[::-1]
        fpr = [(pl >= t).mean() for t in thr] + [1.0]
        tpr = [(hp >= t).mean() for t in thr] + [1.0]
        ax.plot(fpr, tpr, lw=2, color=["#d64545", "#2b6cb0"][i],
                label=f"{name}  AUC={auc(pl, hp):.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="plain-text FPR", ylabel="honeypot hit rate",
           title="ROC: honeypot vs plain controls\n(bias-free discrimination)")
    ax.legend(loc="lower right")

    ax = axes[2]  # zoomed low-FPR region — the operating regime that matters
    for i, (name, (plain, honey)) in enumerate(data.items()):
        pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
        thr = np.unique(np.concatenate([pl, hp]))[::-1]
        fpr = [(pl >= t).mean() for t in thr] + [1.0]
        tpr = [(hp >= t).mean() for t in thr] + [1.0]
        ax.plot(fpr, tpr, lw=2, marker="o", ms=3, color=["#d64545", "#2b6cb0"][i],
                label=name)
    ax.set(xlim=(-0.002, 0.10), xlabel="plain-text FPR", ylabel="honeypot hit rate",
           title="ROC zoom: FPR ≤ 10%")
    ax.legend(loc="lower right")

    fig.tight_layout()
    png = HERE / "eval_awareness_bias.png"
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
