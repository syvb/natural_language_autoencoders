"""Front-loading: compare the SAME steering sweep decoded through four verbalizers.

  - v3 NLA   (bullets prompt, U[1,120] token-trunc RL, syvb/nla-qwen2.5-7b-L20-v3-rl
               iter_0000200)                       -> frontload_v3_raw_judged.json
  - v2 NLA   (item-trunc RL)                       -> frontload_v2model_raw_judged.json
  - v1 NLA   (rltrunc-gradguard kl0.01/iter_200)   -> frontload_v2_raw_judged.json
  - kitft    (base verbalizer)                     -> frontload_kitft_raw_judged.json

Identical injected activations across all four (same genuine dirs + neutral bases), so any
difference is the verbalizer's. 3 panels: appearance rate, conditional median first-index
(appearing rows only), and conditional normalized rank. Same layout as plot_model_compare.py.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
MODELS = [
    ("v3 NLA", "frontload_v3_raw_judged.json", "D-", 2.4, 5, 1.0),
    ("v2 NLA", "frontload_v2model_raw_judged.json", "o-", 1.8, 4, 0.75),
    ("v1 NLA", "frontload_v2_raw_judged.json", "^--", 1.6, 4, 0.6),
    ("kitft", "frontload_kitft_raw_judged.json", "s:", 1.4, 3, 0.5),
]
TCOL = {"sycophancy": "#1f77b4", "neuroticism": "#9467bd", "yellow": "#e8b800"}
MISS = 10


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


def build(rows):
    agg = defaultdict(lambda: {"appidx": [], "appnorm": [], "app": 0, "tot": 0})
    pts = defaultdict(lambda: ([], []))
    ptsN = defaultdict(lambda: ([], []))
    ptsA = defaultdict(lambda: ([], []))
    for r in rows:
        fi = r.get("first_index"); appears = bool(fi and fi >= 1)
        a = agg[(r["trait"], r["r"])]; a["tot"] += 1; a["app"] += int(appears)
        ptsA[r["trait"]][0].append(r["r"]); ptsA[r["trait"]][1].append(int(appears))
        if appears:
            nr = fi / max(1, r["n_items"])
            a["appidx"].append(fi); a["appnorm"].append(nr)
            pts[r["trait"]][0].append(r["r"]); pts[r["trait"]][1].append(fi)
            ptsN[r["trait"]][0].append(r["r"]); ptsN[r["trait"]][1].append(nr)
    return agg, pts, ptsN, ptsA


def threshold(agg, trait, rs, thr=0.5):
    for r in rs:
        a = agg[(trait, r)]
        if a["tot"] and a["app"] / a["tot"] >= thr:
            return r
    return float("nan")


fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
summary = []
for name, fn, style, lw, ms, alpha in MODELS:
    path = f"{R}/{fn}"
    if not os.path.exists(path):
        print(f"  [skip] {name}: {fn} missing"); continue
    rows = json.load(open(path))
    agg, pts, ptsN, ptsA = build(rows)
    rs = sorted({k[1] for k in agg})
    med_items = np.median([x["n_items"] for x in rows])
    print(f"\n=== {name} ({fn}, {len(rows)} rows, median {med_items:.0f} items) ===")
    for trait, col in TCOL.items():
        rr = [r for r in rs if (trait, r) in agg]
        if not rr:
            continue
        appr = [agg[(trait, r)]["app"] / agg[(trait, r)]["tot"] for r in rr]
        cond_rr = [r for r in rr if agg[(trait, r)]["appidx"]]
        cond_med = [np.median(agg[(trait, r)]["appidx"]) for r in cond_rr]
        cond_nrm = [np.median(agg[(trait, r)]["appnorm"]) for r in cond_rr]
        sp_app = spearman(*ptsA[trait]); sp_cond = spearman(*pts[trait]); sp_nrm = spearman(*ptsN[trait])
        thr = threshold(agg, trait, rr, 0.5)
        ax[0].plot(rr, appr, style, color=col, lw=lw, ms=ms, alpha=alpha, label=f"{trait} · {name}")
        ax[1].plot(cond_rr, cond_med, style, color=col, lw=lw, ms=ms, alpha=alpha)
        ax[2].plot(cond_rr, cond_nrm, style, color=col, lw=lw, ms=ms, alpha=alpha)
        print(f"  {trait:11s} thr(app>=.5) r={thr if thr==thr else float('nan'):<5}"
              f"  sp(r,appears)={sp_app:+.2f}  sp(r,idx|app)={sp_cond:+.2f}  sp(r,normrank|app)={sp_nrm:+.2f}")
        summary.append((name, trait, med_items, thr, sp_app, sp_cond, sp_nrm))

for a in ax:
    a.set_xscale("log"); a.set_xlabel("steering strength r (log)"); a.grid(alpha=.3)
ax[0].set_ylabel("appearance rate"); ax[0].set_title("Trait appears in the list"); ax[0].set_ylim(-0.03, 1.03)
ax[0].legend(fontsize=7, ncol=1, loc="upper left")
ax[1].set_ylabel("median first-index | appears"); ax[1].set_title("Conditional absolute rank (climbs to top)"); ax[1].set_yscale("log"); ax[1].invert_yaxis()
ax[2].set_ylabel("median normalized rank | appears"); ax[2].set_title("Conditional normalized rank (length-robust)"); ax[2].set_ylim(1.02, -0.02)
fig.suptitle("Front-loading: same steering sweep, four verbalizers  (diamond=v3, circle=v2, dashed=v1, dotted=kitft)",
             y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(f"{R}/fig_frontload_model_compare_v3.png", dpi=130, bbox_inches="tight")
print("\nwrote results/fig_frontload_model_compare_v3.png")
print("\n=== SUMMARY ===")
print(f"{'model':8s} {'trait':11s} {'med_items':>9s} {'thr@.5':>7s} {'sp(app)':>8s} {'sp(idx|app)':>11s} {'sp(nrm|app)':>11s}")
for name, trait, mi, thr, sa, sc, sn in summary:
    print(f"{name:8s} {trait:11s} {mi:>9.0f} {thr if thr==thr else -1:>7.2f} {sa:>8.2f} {sc:>11.2f} {sn:>11.2f}")
