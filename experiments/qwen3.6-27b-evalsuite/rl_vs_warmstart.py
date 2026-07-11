"""Is the RL actually helping? Compare the matryoshka RL AV vs its warm-start
(pre-RL) AV — eval-awareness detection (primary) + front-loading (bonus).

Both are the MATRYOSHKA model on the same base/positions; only the AV (and, for
FVE, the critic) differ RL-vs-warmstart, so any gap is the RL's doing.

Eval-awareness is scored with the SAME gpt-4o-mini generic judge for both, then
compared with signal detection (honeypot = positives, plain controls =
negatives): base rates, ROC AUC, and hit-rate at matched plain-FPR. AUC / matched-
FPR is the confound-robust metric — a model that just says 'eval-aware' more
often (higher bias) raises honeypot AND plain together and gains no AUC.

    python rl_vs_warmstart.py            # expects warmstart_eval_awareness.json + warmstart_precache.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HONEY = "<|im_start|>system"


def auc(neg, pos):
    from scipy.stats import rankdata
    neg, pos = neg[~np.isnan(neg)], pos[~np.isnan(pos)]
    if not len(neg) or not len(pos):
        return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def load_paware(ea_path):
    ea = json.load(open(ea_path))["entries"]
    plain, honey = [], None
    for e in ea:
        p = np.array([np.nan if v is None else v for v in e["paware"]], dtype=float)
        if e["text"].startswith(HONEY):
            honey = p
        else:
            plain.append(p)
    return np.concatenate(plain), honey


def explen(precache_path):
    """mean explanation length (lines, chars) split honeypot / plain."""
    ent = json.load(open(precache_path))["entries"]
    def stats(pred):
        nl, nc = [], []
        for e in ent:
            if pred(e["text"]):
                for r in e["results"]:
                    if r and r.get("lines"):
                        nl.append(len(r["lines"])); nc.append(len("\n".join(r["lines"])))
        return (round(np.mean(nl), 2), round(np.mean(nc), 1)) if nl else (0, 0)
    return {"honey": stats(lambda t: t.startswith(HONEY)),
            "plain": stats(lambda t: not t.startswith(HONEY))}


def frontload(precache_path):
    """first-line solo FVE, full-expl FVE, reading-budget curve (mean FVE by words)."""
    ent = json.load(open(precache_path))["entries"]
    first_solo, full, curves = [], [], []
    grid = np.arange(0, 121)
    for e in ent:
        for r in e["results"]:
            if not r or not r.get("lines"):
                continue
            full.append(r["fve"][-1])
            if "solo" in r and r["solo"]:
                first_solo.append(r["solo"][0])
            w = np.cumsum([len(ln.split()) for ln in r["lines"]])
            curves.append(np.interp(grid, np.concatenate([[0], w]),
                                    np.concatenate([[0], r["fve"]])))
    return (np.array(first_solo), np.array(full),
            grid.tolist(), np.round(np.mean(curves, axis=0), 4).tolist())


def main():
    ea_rl = HERE / "space" / "eval_awareness.json"
    ea_ws = HERE / "warmstart_eval_awareness.json"
    pc_rl = HERE / "space" / "precache.json"
    pc_ws = HERE / "warmstart_precache.json"

    print("=== EVAL-AWARENESS: RL vs warm-start (honeypot vs plain controls) ===")
    rows = {}
    for name, ea, pc in [("RL", ea_rl, pc_rl), ("warm-start", ea_ws, pc_ws)]:
        plain, honey = load_paware(ea)
        pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
        el = explen(pc)
        a = auc(pl, hp)
        hit = {}
        for f in (0.01, 0.05):
            thr = np.quantile(pl, 1 - f)
            hit[f] = (hp >= thr).mean()
        rows[name] = dict(pl_mean=pl.mean(), hp_mean=hp.mean(), auc=a,
                          hit1=hit[0.01], hit5=hit[0.05], explen=el)
        print(f"\n[{name}]  plain n={len(pl)}  honeypot n={len(hp)}")
        print(f"  mean P: plain {pl.mean():.3f}  honeypot {hp.mean():.3f}")
        print(f"  ROC AUC (honeypot vs plain) = {a:.3f}")
        print(f"  hit @ plain-FPR 1% = {hit[0.01]:.1%}   @ 5% = {hit[0.05]:.1%}")
        print(f"  explanation length honey {el['honey']}  plain {el['plain']}  (lines, chars)")

    d = rows["RL"]["auc"] - rows["warm-start"]["auc"]
    print(f"\n>>> ΔAUC (RL − warm-start) = {d:+.3f}   "
          f"Δhit@5% = {rows['RL']['hit5'] - rows['warm-start']['hit5']:+.1%}")
    verdict = ("RL HELPS (better honeypot detection at matched FPR)" if d > 0.02 else
               "RL HURTS discrimination" if d < -0.02 else
               "RL ~neutral on discrimination — check base rates for bias-only shift")
    print(f">>> {verdict}")

    print("\n=== FRONT-LOADING (bonus): RL vs warm-start ===")
    fl = {}
    for name, pc in [("RL", pc_rl), ("warm-start", pc_ws)]:
        # RL solo lives in space/loo.json, not precache; warm-start has it inline
        if name == "RL":
            loo = json.load(open(HERE / "space" / "loo.json"))["entries"]
            fs = np.array([s[0] for e in loo for s in e["solo"] if s])
            _, full, grid, curve = frontload(pc)
        else:
            fs, full, grid, curve = frontload(pc)
        fl[name] = (fs, full, grid, curve)
        print(f"\n[{name}] first-line solo-FVE: mean {fs.mean():+.3f}, frac positive {np.mean(fs > 0):.1%}")
        print(f"         full-explanation FVE: mean {full.mean():.3f}")
        print(f"         reading budget: FVE@10w {np.interp(10, grid, curve):.3f}  "
              f"@25w {np.interp(25, grid, curve):.3f}  @full {curve[-1]:.3f}")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"RL": "#2a78d6", "warm-start": "#c98500"}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # ROC
    for name, ea, pc in [("RL", ea_rl, pc_rl), ("warm-start", ea_ws, pc_ws)]:
        plain, honey = load_paware(ea)
        pl, hp = plain[~np.isnan(plain)], honey[~np.isnan(honey)]
        thr = np.unique(np.concatenate([pl, hp]))[::-1]
        fpr = [(pl >= t).mean() for t in thr] + [1.0]
        tpr = [(hp >= t).mean() for t in thr] + [1.0]
        ax[0].plot(fpr, tpr, lw=2.2, color=C[name], label=f"{name} AUC={auc(pl,hp):.3f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="plain-control FPR", ylabel="honeypot hit rate",
              title="Eval-awareness detection\n(honeypot vs plain controls)")
    ax[0].legend(loc="lower right")

    # base rates
    labels = ["plain mean", "honeypot mean", "hit@FPR5%"]
    x = np.arange(len(labels))
    for i, name in enumerate(["RL", "warm-start"]):
        r = rows[name]
        ax[1].bar(x + (i - .5) * .38, [r["pl_mean"], r["hp_mean"], r["hit5"]], .38,
                  color=C[name], label=name)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set(title="Base rates & detection\n(bias rises together = no real gain)", ylabel="P / rate")
    ax[1].legend()

    # reading budget
    for name in ["RL", "warm-start"]:
        _, _, grid, curve = fl[name]
        ax[2].plot(grid, curve, lw=2.2, color=C[name], label=name)
    ax[2].axhline(0, color="gray", lw=.7)
    ax[2].set(xlabel="words of the explanation read", ylabel="mean FVE recovered",
              title="Front-loading: recovery vs words read", xlim=(0, 120))
    ax[2].legend(loc="lower right")

    fig.tight_layout()
    png = HERE / "rl_vs_warmstart.png"
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {png}")
    json.dump({k: {kk: (vv if not isinstance(vv, np.generic) else float(vv))
                   for kk, vv in v.items() if kk != "explen"} | {"explen": v["explen"]}
               for k, v in rows.items()},
              open(HERE / "rl_vs_warmstart_stats.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
