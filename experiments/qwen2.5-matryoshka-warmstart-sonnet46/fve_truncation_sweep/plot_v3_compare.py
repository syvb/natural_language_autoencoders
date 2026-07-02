"""FVE-vs-truncation-length: v3 RL NLA vs v2, v1, kitft, and the v3 warm-start.

Reads results/*.csv (token_fve_v3.csv = v3 iter_200; token_fve_v3ws.csv = v3
pre-RL warm-start). Token panels are the apples-to-apples unit. Local, matplotlib.
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

# (filename-suffix, label, color, lw, marker)
ARMS = [
    ("_v3", "v3 RL (iter_200, bullets prompt, U[1,120])", "#9467bd", 2.6, None),
    ("_v2", "v2 RL (iter_200, item-trunc)", "#2ca02c", 1.8, None),
    ("",    "v1 RLed trunc (kl0.01/iter_200)", "#1f77b4", 1.8, None),
    ("_kitft", "kitft (control, non-trunc RL)", "#888888", 1.6, "s"),
    ("_v3ws", "v3 warm-start (pre-RL SFT)", "#ff7f0e", 1.6, "^"),
]


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
for suf, label, color, lw, mk in ARMS:
    tL, tF = load(f"token_fve{suf}.csv", XCAP)
    fL, fF = load(f"token_fve{suf}.csv")
    kL, kF = load(f"lines_fve{suf}.csv")
    if tL is None:
        print(f"  [skip] {label}: token_fve{suf}.csv missing"); continue
    arms.append(dict(suf=suf, label=label, color=color, lw=lw, mk=mk, tL=tL, tF=tF, kL=kL, kF=kF,
                     full=float(fF[-1]), peak=float(fF.max()), peak_at=int(fL[int(fF.argmax())])))

print(f"{'arm':46s} {'tok>=0':>7s} {'tok>=.5':>8s} {'full':>6s} {'peak(@tok)':>12s}")
for d in arms:
    print(f"{d['label']:46s} {str(thresh(d['tL'],d['tF'],0)):>7s} {str(thresh(d['tL'],d['tF'],0.5)):>8s} "
          f"{d['full']:>6.3f}  {d['peak']:.3f}@{d['peak_at']}")

# ---- main: token FVE (linear) + line FVE ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
for d in arms:
    a1.plot(d["tL"], d["tF"], color=d["color"], lw=d["lw"],
            label=f"{d['label']}  (full={d['full']:.3f})")
a1.axhline(0, color="k", lw=.6, alpha=.4); a1.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a1.set_xlabel("AV explanation truncation length (content tokens)")
a1.set_ylabel("round-trip FVE"); a1.set_xlim(0, XCAP)
a1.set_title("FVE vs token-truncation length"); a1.grid(alpha=.3); a1.legend(loc="lower right", fontsize=8.5)
for d in arms:
    a2.plot(d["kL"], d["kF"], marker=d["mk"] or "o", ms=4, color=d["color"], lw=d["lw"])
a2.axhline(0, color="k", lw=.6, alpha=.4); a2.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a2.set_xlabel("AV explanation truncation length (lines)")
a2.set_ylabel("round-trip FVE"); a2.set_xlim(0, 40)
a2.set_title("FVE vs line-truncation length (per-model; not cross-comparable)")
a2.grid(alpha=.3)
fig.suptitle("Round-trip FVE vs AV-explanation truncation — v3 RL NLA vs v2/v1/kitft", y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_v3_compare.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- v3-only single view (only when the v3 arm is actually present) ----
d = arms[0] if (arms and arms[0]["suf"] == "_v3") else None
if d is None:
    print("  [skip] v3-only view: _v3 CSVs missing")
    raise SystemExit(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(d["tL"], d["tF"], lw=2, color="#9467bd")
a1.axhline(d["full"], ls="--", color="gray", lw=1, label=f"full-len FVE={d['full']:.3f}")
a1.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a1.set_xlabel("truncation length (content tokens)"); a1.set_ylabel("round-trip FVE")
a1.set_xlim(0, XCAP); a1.set_title("FVE vs token-truncation length"); a1.grid(alpha=.3); a1.legend()
a2.plot(d["kL"], d["kF"], marker="o", ms=4, lw=2, color="#9467bd")
a2.axhline(d["full"], ls="--", color="gray", lw=1, label=f"full-len FVE={d['full']:.3f}")
a2.set_xlabel("truncation length (lines)"); a2.set_ylabel("round-trip FVE")
a2.set_xlim(0, 40); a2.set_title("FVE vs line-truncation length"); a2.grid(alpha=.3); a2.legend()
fig.suptitle(f"v3 RL NLA — round-trip FVE vs AV-explanation truncation ({d['label']})", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_v3_only.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- log-log: unexplained variance (1 - FVE) vs truncation length ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
for d in arms:
    a1.loglog(d["tL"], 1 - d["tF"], color=d["color"], lw=d["lw"], label=d["label"])
for d in arms:
    a1.axhline(1 - d["full"], ls="--", color=d["color"], lw=1, alpha=.55)
a1.set_xlabel("AV explanation truncation length (content tokens, log)")
a1.set_ylabel("unexplained variance  1 − FVE  (log)")
a1.set_xlim(right=XCAP); a1.set_title("Reconstruction error vs token-truncation length")
a1.grid(which="both", alpha=.3); a1.legend(loc="lower left", fontsize=8.5)
for d in arms:
    a2.loglog(d["kL"], 1 - d["kF"], marker=d["mk"] or "o", ms=4, color=d["color"], lw=d["lw"], label=d["label"])
for d in arms:
    a2.axhline(1 - d["full"], ls="--", color=d["color"], lw=1, alpha=.55)
a2.set_xlabel("AV explanation truncation length (lines, log)")
a2.set_ylabel("unexplained variance  1 − FVE  (log)")
a2.set_title("Reconstruction error vs line-truncation length (per-model)")
a2.grid(which="both", alpha=.3)
fig.suptitle("log-log reconstruction error (1 − FVE) vs AV-explanation truncation — v3 vs v2/v1/kitft",
             y=1.02, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fve_truncation_v3_compare_loglog.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("\nwrote fve_truncation_v3_compare.png + fve_truncation_v3_only.png + fve_truncation_v3_compare_loglog.png")
