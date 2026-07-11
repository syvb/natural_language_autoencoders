"""Three-way: matryoshka RL vs matryoshka warm-start vs standard (normal) RL.

All eval-awareness scores already exist (same gpt-4o-mini judge, same 5,275
positions). Disentangles two axes:
  - RL vs warm-start (does the RL training help?)
  - matryoshka vs standard format (does the ranked-list format help?)

    matryoshka RL   : space/{eval_awareness,precache,loo}.json     (line units)
    matryoshka warm : warmstart_{eval_awareness,precache}.json     (line units, solo inline)
    standard RL     : space_std/{eval_awareness,precache,loo}.json (sentence units)

    python three_way.py
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HONEY = "<|im_start|>system"
SENT = re.compile(r"(?<=[.!?])\s+")

MODELS = {
    "matryoshka RL":   dict(ea="space/eval_awareness.json",   pc="space/precache.json",
                            loo="space/loo.json",   unit="line",  c="#2a78d6"),
    "matryoshka warm": dict(ea="warmstart_eval_awareness.json", pc="warmstart_precache.json",
                            loo=None,                unit="line",  c="#c98500"),
    "standard RL":     dict(ea="space_std/eval_awareness.json", pc="space_std/precache.json",
                            loo="space_std/loo.json", unit="sent", c="#e34948"),
}


def auc(neg, pos):
    from scipy.stats import rankdata
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    r = rankdata(np.concatenate([neg, pos]))
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def paware(ea_path):
    ea = json.load(open(HERE / ea_path))["entries"]
    plain, honey = [], None
    for e in ea:
        p = np.array([np.nan if v is None else v for v in e["paware"]], dtype=float)
        if e["text"].startswith(HONEY):
            honey = p
        else:
            plain.append(p)
    pl = np.concatenate(plain); pl = pl[~np.isnan(pl)]
    hp = honey[~np.isnan(honey)]
    return pl, hp


def frontload(cfg):
    """first-unit-solo array, full-FVE array."""
    pc = json.load(open(HERE / cfg["pc"]))["entries"]
    if cfg["loo"]:
        loo = json.load(open(HERE / cfg["loo"]))["entries"]
        first_solo = np.array([s[0] for e in loo for s in e["solo"] if s])
    else:  # warm-start: solo inline in precache
        first_solo = np.array([r["solo"][0] for e in pc for r in e["results"]
                               if r and r.get("solo")])
    full = np.array([r["fve"][-1] for e in pc for r in e["results"]
                     if r and r.get("fve")])
    return first_solo, full


def main():
    print(f"{'model':17s} {'AUC':>6s} {'plainμ':>7s} {'honeyμ':>7s} {'hit@1%':>7s} {'hit@5%':>7s} "
          f"| {'1st-solo+':>9s} {'fullFVE':>7s}")
    stats = {}
    for name, cfg in MODELS.items():
        pl, hp = paware(cfg["ea"])
        a = auc(pl, hp)
        h1 = (hp >= np.quantile(pl, 0.99)).mean()
        h5 = (hp >= np.quantile(pl, 0.95)).mean()
        fs, full = frontload(cfg)
        stats[name] = dict(auc=a, plain=pl.mean(), honey=hp.mean(), h1=h1, h5=h5,
                           solo_pos=np.mean(fs > 0), full=full.mean(), pl=pl, hp=hp)
        print(f"{name:17s} {a:6.3f} {pl.mean():7.3f} {hp.mean():7.3f} {h1:7.1%} {h5:7.1%} "
              f"| {np.mean(fs > 0):9.1%} {full.mean():7.3f}")

    print("\ninterpretation:")
    print("  eval-awareness AUC: RL >> warm-start (RL creates the signal); "
          "matryoshka RL vs standard RL similar")
    print("  hit@matched-FPR: standard RL highest (less saturated scores; bias confound)")
    print("  front-loading (1st-unit-useful-alone): matryoshka RL unique — needs BOTH "
          "format AND RL")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # ROC
    for name, cfg in MODELS.items():
        s = stats[name]
        thr = np.unique(np.concatenate([s["pl"], s["hp"]]))[::-1]
        fpr = [(s["pl"] >= t).mean() for t in thr] + [1.0]
        tpr = [(s["hp"] >= t).mean() for t in thr] + [1.0]
        ax[0].plot(fpr, tpr, lw=2.2, color=cfg["c"], label=f"{name}  AUC={s['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="plain-control FPR", ylabel="honeypot hit rate",
              title="Eval-awareness detection")
    ax[0].legend(loc="lower right", fontsize=9)

    # eval-awareness bars: AUC + hit@5%
    names = list(MODELS)
    x = np.arange(len(names))
    ax[1].bar(x - 0.19, [stats[n]["auc"] for n in names], 0.38,
              color=[MODELS[n]["c"] for n in names], label="AUC")
    ax[1].bar(x + 0.19, [stats[n]["h5"] for n in names], 0.38,
              color=[MODELS[n]["c"] for n in names], alpha=.5, label="hit@FPR5%")
    ax[1].axhline(0.5, color="gray", lw=.7, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
    ax[1].set(title="Eval-awareness: AUC (solid) & hit@5% (faded)\ndotted = chance AUC", ylabel="")
    ax[1].legend(fontsize=8)

    # front-loading bars: 1st-unit-solo-positive% + full-FVE
    ax[2].bar(x - 0.19, [stats[n]["solo_pos"] for n in names], 0.38,
              color=[MODELS[n]["c"] for n in names], label="1st unit useful alone")
    ax[2].bar(x + 0.19, [stats[n]["full"] for n in names], 0.38,
              color=[MODELS[n]["c"] for n in names], alpha=.5, label="full-expl FVE")
    ax[2].set_xticks(x); ax[2].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
    ax[2].set(title="Front-loading: 1st-unit-alone useful (solid)\n& full-explanation FVE (faded)", ylabel="")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    png = HERE / "three_way.png"
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
