"""Round-trip FVE vs AV-explanation truncation length across the 4 v2 RL checkpoints
(iter 50/100/150/200), each AV paired with its matched co-trained critic.

Reads results/token_fve_iter{IT}.csv / lines_fve_iter{IT}.csv. Produces a linear panel-pair
(token + line FVE) and a log-log panel-pair (unexplained variance 1-FVE). Shows how the
round-trip improves and front-loads over RL training. Local, matplotlib only.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
XCAP = 160
ITERS = ["0000050", "0000100", "0000150", "0000200"]
CMAP = plt.get_cmap("viridis")
COLORS = {it: CMAP(i / (len(ITERS) - 1) * 0.9) for i, it in enumerate(ITERS)}


def load(name, cap=None):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return None, None
    L, F = [], []
    for r in csv.DictReader(open(path)):
        li = int(r[list(r)[0]])
        if cap is not None and li > cap:
            continue
        L.append(li); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


def thresh(L, F, t):
    for li, f in zip(L, F):
        if f >= t:
            return int(li)
    return None


arms = []
for it in ITERS:
    tL, tF = load(f"token_fve_iter{it}.csv", XCAP)
    fL, fF = load(f"token_fve_iter{it}.csv")
    kL, kF = load(f"lines_fve_iter{it}.csv")
    if tL is None:
        print(f"  [skip] iter_{it}: csv missing"); continue
    arms.append(dict(it=it, label=f"iter {int(it)}", color=COLORS[it], tL=tL, tF=tF, kL=kL, kF=kF,
                     full=float(fF[-1]), peak=float(fF.max()), peak_at=int(fL[int(fF.argmax())])))

print(f"{'ckpt':10s} {'tok>=0':>7s} {'tok>=.5':>8s} {'peak(@tok)':>12s} {'full':>6s}")
for d in arms:
    print(f"{d['label']:10s} {str(thresh(d['tL'],d['tF'],0)):>7s} {str(thresh(d['tL'],d['tF'],0.5)):>8s} "
          f"{d['peak']:.3f}@{d['peak_at']:<4d} {d['full']:>6.3f}")

# ---- linear: token + line FVE ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
for d in arms:
    a1.plot(d["tL"], d["tF"], color=d["color"], lw=2.2, label=f"{d['label']}  (full={d['full']:.3f})")
a1.axhline(0, color="k", lw=.6, alpha=.4); a1.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a1.set_xlabel("AV explanation truncation length (content tokens)"); a1.set_ylabel("round-trip FVE")
a1.set_xlim(0, XCAP); a1.set_title("FVE vs token-truncation length"); a1.grid(alpha=.3); a1.legend(loc="lower right", fontsize=9)
for d in arms:
    a2.plot(d["kL"], d["kF"], marker="o", ms=3.5, color=d["color"], lw=2.2, label=d["label"])
a2.axhline(0, color="k", lw=.6, alpha=.4); a2.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a2.set_xlabel("AV explanation truncation length (lines)"); a2.set_ylabel("round-trip FVE")
a2.set_xlim(0, 40); a2.set_title("FVE vs line-truncation length"); a2.grid(alpha=.3); a2.legend(loc="lower right", fontsize=9)
fig.suptitle("Round-trip FVE vs AV-explanation truncation — 4 v2 RL checkpoints (AV+matched critic)", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_ckpts_compare.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- log-log: unexplained variance 1-FVE ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
for d in arms:
    a1.loglog(d["tL"], 1 - d["tF"], color=d["color"], lw=2.2, label=d["label"])
for d in arms:
    a1.axhline(1 - d["full"], ls="--", color=d["color"], lw=1, alpha=.5)
a1.set_xlabel("AV explanation truncation length (content tokens, log)")
a1.set_ylabel("unexplained variance  1 − FVE  (log)")
a1.set_xlim(right=XCAP); a1.set_title("Reconstruction error vs token-truncation length")
a1.grid(which="both", alpha=.3); a1.legend(loc="lower left", fontsize=9)
for d in arms:
    a2.loglog(d["kL"], 1 - d["kF"], marker="o", ms=3.5, color=d["color"], lw=2.2, label=d["label"])
for d in arms:
    a2.axhline(1 - d["full"], ls="--", color=d["color"], lw=1, alpha=.5)
a2.set_xlabel("AV explanation truncation length (lines, log)")
a2.set_ylabel("unexplained variance  1 − FVE  (log)")
a2.set_title("Reconstruction error vs line-truncation length"); a2.grid(which="both", alpha=.3); a2.legend(loc="lower left", fontsize=9)
fig.suptitle("log-log reconstruction error (1 − FVE) vs AV-explanation truncation — 4 v2 RL checkpoints", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_ckpts_compare_loglog.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("\nwrote fve_truncation_ckpts_compare.png + fve_truncation_ckpts_compare_loglog.png")
