"""Regenerate all FVE-vs-truncation figures from results/*.csv (run locally, CPU).

Three arms (control + warm-start are optional; plotted only if their CSVs exist):
  - RLed truncation  : token_fve.csv          / lines_fve.csv
  - published kitft  : token_fve_kitft.csv    / lines_fve_kitft.csv   (control)
  - warm-start pre-RL: token_fve_warmstart.csv/ lines_fve_warmstart.csv

Pull CSVs from a box into results/ first (see README), then: python plot.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
XCAP = 150  # tokens past ~150 are rare; cap the x-axis there


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


# arm: (key, label, color, marker)
ARMS = [
    ("warmstart", "warm-start (pre-RL)", "#ff7f0e", "^"),
    ("",          "RLed truncation (kl0.01/iter200)", "#1f77b4", None),
    ("kitft",     "published kitft (control)", "#888888", "s"),
]


def suffix(key):
    return f"_{key}" if key else ""


def save(fig, name):
    fig.savefig(os.path.join(R, name), dpi=140, bbox_inches="tight"); plt.close(fig)


# load all arms
data = {}
for key, label, color, mk in ARMS:
    tL, tF = load(f"token_fve{suffix(key)}.csv", XCAP)
    kL, kF = load(f"lines_fve{suffix(key)}.csv")
    if tL is None:
        continue
    data[key] = dict(label=label, color=color, mk=mk, tL=tL, tF=tF, kL=kL, kF=kF,
                     full=tF[-1])
present = list(data.values())
FULL_RL = data[""]["full"]

# ---------- standalone token FVE (all arms) ----------
fig, a = plt.subplots(figsize=(8, 5.4))
for d in present:
    a.plot(d["tL"], d["tF"], color=d["color"], lw=2, label=f"{d['label']}  (full={d['full']:.3f})")
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a.set_xlabel("AV explanation truncation length (content tokens)"); a.set_ylabel("round-trip FVE")
a.set_title("Round-trip FVE vs token-truncation length (N=100)")
a.set_xlim(0, XCAP); a.grid(alpha=.3); a.legend(loc="lower right", fontsize=9)
fig.tight_layout(); save(fig, "standalone_token_fve_compare.png")

# ---------- standalone token log-log error (all arms) ----------
fig, a = plt.subplots(figsize=(8, 5.4))
for d in present:
    a.loglog(d["tL"], 1 - d["tF"], color=d["color"], lw=2, label=d["label"])
for d in present:
    a.axhline(1 - d["full"], ls="--", color=d["color"], lw=1, alpha=.6)
a.set_xlabel("AV explanation truncation length (content tokens, log)")
a.set_ylabel("unexplained variance  1 − FVE  (log)")
a.set_title("Reconstruction error vs token-truncation length (N=100)")
a.set_xlim(right=XCAP); a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=9)
fig.tight_layout(); save(fig, "standalone_token_loglog.png")

# ---------- standalone line FVE (all arms) ----------
fig, a = plt.subplots(figsize=(8, 5.4))
for d in present:
    a.plot(d["kL"], d["kF"], marker=d["mk"] or "o", ms=4, color=d["color"], lw=2, label=d["label"])
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
a.set_xlabel("AV explanation truncation length (lines)"); a.set_ylabel("round-trip FVE")
a.set_title("Round-trip FVE vs line-truncation length (N=100)")
a.grid(alpha=.3); a.legend(loc="lower right", fontsize=9)
fig.tight_layout(); save(fig, "standalone_lines_fve_compare.png")

# ---------- 2x2 overview (tokens/lines x linear/loglog) ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
a = ax[0, 0]
for d in present:
    a.plot(d["tL"], d["tF"], color=d["color"], lw=1.8, label=d["label"])
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.6, ls=":", alpha=.6)
a.set_xlabel("truncation length (content tokens)"); a.set_ylabel("round-trip FVE")
a.set_title("FVE vs token-truncation length"); a.set_xlim(0, XCAP); a.grid(alpha=.3); a.legend(loc="lower right", fontsize=8)
a = ax[0, 1]
for d in present:
    a.plot(d["kL"], d["kF"], marker=d["mk"] or "o", ms=4, color=d["color"], lw=1.8, label=d["label"])
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.6, ls=":", alpha=.6)
a.set_xlabel("truncation length (lines)"); a.set_ylabel("round-trip FVE")
a.set_title("FVE vs line-truncation length"); a.grid(alpha=.3); a.legend(loc="lower right", fontsize=8)
a = ax[1, 0]
for d in present:
    a.loglog(d["tL"], 1 - d["tF"], color=d["color"], lw=1.8, label=d["label"])
for d in present:
    a.axhline(1 - d["full"], color=d["color"], ls="--", lw=1, alpha=.6)
a.set_xlabel("truncation length (content tokens, log)"); a.set_ylabel("unexplained variance 1−FVE (log)")
a.set_title("log-log reconstruction error (tokens)"); a.set_xlim(right=XCAP)
a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=8)
a = ax[1, 1]
for d in present:
    a.loglog(d["kL"], 1 - d["kF"], marker=d["mk"] or "o", ms=4, color=d["color"], lw=1.8, label=d["label"])
for d in present:
    a.axhline(1 - d["full"], color=d["color"], ls="--", lw=1, alpha=.6)
a.set_xlabel("truncation length (lines, log)"); a.set_ylabel("unexplained variance 1−FVE (log)")
a.set_title("log-log reconstruction error (lines)"); a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=8)
fig.suptitle("Round-trip FVE vs AV-explanation truncation — warm-start vs RLed truncation vs published kitft (N=100, same docs)", y=1.0, fontsize=13)
fig.tight_layout(); save(fig, "fve_compare_rl_vs_control.png")

# ---------- RLed-only single-model views ----------
d = data[""]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(d["tL"], d["tF"], lw=1.6, color=d["color"])
a1.axhline(FULL_RL, ls="--", color="gray", lw=1, label=f"full-len FVE={FULL_RL:.3f}")
a1.set_xlabel("truncation length (content tokens)"); a1.set_ylabel("round-trip FVE")
a1.set_title("FVE vs token-truncation length (N=100)"); a1.set_xlim(0, XCAP); a1.grid(alpha=.3); a1.legend()
a2.plot(d["kL"], d["kF"], marker="o", ms=4, lw=1.6, color="#d62728")
a2.axhline(FULL_RL, ls="--", color="gray", lw=1, label=f"full-len FVE={FULL_RL:.3f}")
a2.set_xlabel("truncation length (lines)"); a2.set_ylabel("round-trip FVE")
a2.set_title("FVE vs line-truncation length (N=100)"); a2.grid(alpha=.3); a2.legend()
fig.suptitle("kl0.01/iter_0000200 — round-trip FVE vs AV-explanation truncation", y=1.02)
fig.tight_layout(); save(fig, "fve_vs_truncation.png")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.loglog(d["tL"], 1 - d["tF"], lw=1.6, color=d["color"])
a1.axhline(1 - FULL_RL, ls="--", color="gray", lw=1, label=f"floor 1−FVE={1-FULL_RL:.3f}")
a1.set_xlabel("truncation length (content tokens, log)"); a1.set_ylabel("unexplained variance 1−FVE (log)")
a1.set_title("Reconstruction error vs token-truncation (N=100)"); a1.set_xlim(right=XCAP)
a1.grid(which="both", alpha=.3); a1.legend()
a2.loglog(d["kL"], 1 - d["kF"], marker="o", ms=4, lw=1.6, color="#d62728")
a2.axhline(1 - FULL_RL, ls="--", color="gray", lw=1, label=f"floor 1−FVE={1-FULL_RL:.3f}")
a2.set_xlabel("truncation length (lines, log)"); a2.set_ylabel("unexplained variance 1−FVE (log)")
a2.set_title("Reconstruction error vs line-truncation (N=100)"); a2.grid(which="both", alpha=.3); a2.legend()
fig.suptitle("kl0.01/iter_0000200 — log-log reconstruction error (1−FVE)", y=1.02)
fig.tight_layout(); save(fig, "fve_vs_truncation_loglog.png")

print(f"wrote figures to {R}  ({len(present)} arms: {', '.join(d['label'] for d in present)})")
