"""Post-ready figures: the matryoshka NLA vs the kitft baseline only.

Reads the temperature-1 regenerated data in this directory:
  token_fve_t1_iter0000{050,100,150,200}.csv / lines_* , token_fve_t1_kitft.csv
  frontload_v3_t1_raw_judged.json / frontload_kitft_t1_raw_judged.json
Writes fig1..fig4*. Neutral labels (no version names, no sampling labels).
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = "matryoshka NLA (ours)"
BASE = "kitft baseline"
C_OURS, C_BASE = "#9467bd", "#888888"
XCAP = 160


def load(name, cap=None):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None, None
    L, F = [], []
    for r in csv.DictReader(open(path)):
        li = int(r[list(r)[0]])
        if cap is not None and li > cap:
            continue
        L.append(li); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


# ---- fig1: FVE vs truncation (linear + loglog) ----
tL, tF = load("token_fve_t1_iter0000200.csv", XCAP)
kL, kF = load("token_fve_t1_kitft.csv", XCAP)
if tL is not None and kL is not None:
    _, tFfull = load("token_fve_t1_iter0000200.csv")
    _, kFfull = load("token_fve_t1_kitft.csv")
    tLf, tFf = load("token_fve_t1_iter0000200.csv")
    kLf, kFf = load("token_fve_t1_kitft.csv")
    fig, a1 = plt.subplots(figsize=(8.2, 5.2))
    a1.plot(tLf, tFf, color=C_OURS, lw=2.6, label=f"{OURS}  (full={tFfull[-1]:.3f})")
    a1.plot(kLf, kFf, color=C_BASE, lw=2.0, ls="--", label=f"{BASE}  (full={kFfull[-1]:.3f})")
    a1.axhline(0, color="k", lw=.6, alpha=.4); a1.axhline(0.5, color="green", lw=.7, ls=":", alpha=.6)
    a1.set_xlabel("explanation truncation length (content tokens)")
    a1.set_ylabel("round-trip FVE"); a1.set_xlim(0, kLf.max())  # x exactly spans the kitft curve
    a1.set_title("Reconstruction quality vs truncation length")
    a1.grid(alpha=.3); a1.legend(loc="lower right", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig1a_fve_truncation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, a2 = plt.subplots(figsize=(8.2, 5.2))
    a2.loglog(tL, 1 - tF, color=C_OURS, lw=2.6, label=OURS)
    a2.loglog(kL, 1 - kF, color=C_BASE, lw=2.0, ls="--", label=BASE)
    a2.set_xlabel("truncation length (content tokens, log)")
    a2.set_ylabel("unexplained variance  1 − FVE  (log)")
    a2.set_xlim(right=XCAP); a2.set_title("Reconstruction error vs truncation length (log-log)")
    a2.grid(which="both", alpha=.3); a2.legend(loc="lower left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig1b_fve_truncation_loglog.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig1a/fig1b ok")

# ---- fig2: marginal ΔFVE per token ----
if tL is not None and kL is not None:
    WIN = 5
    def marg(F):
        m = np.diff(F, prepend=0.0)
        return np.convolve(m, np.ones(WIN) / WIN, mode="same")
    tLm, tFm = load("token_fve_t1_iter0000200.csv", 120)
    kLm, kFm = load("token_fve_t1_kitft.csv", 120)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(tLm, marg(tFm), color=C_OURS, lw=2.4, label=OURS)
    ax.plot(kLm, marg(kFm), color=C_BASE, lw=2.0, ls="--", label=BASE)
    ax.axhline(0, color="k", lw=.8, alpha=.5)
    ax.set_xlim(0, 120)
    ax.set_xlabel("token index in explanation")
    ax.set_ylabel(f"additional FVE per token (ΔFVE, {WIN}-token rolling mean)")
    ax.set_title("Marginal variance explained per token")
    ax.grid(alpha=.3); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig2_marginal_pertoken.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig2 ok")

# ---- fig3: training progression ----
ITERS = ["0000050", "0000100", "0000150", "0000200"]
CMAP = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(8.8, 5.2))
any_it = False
wL, wF = load("token_fve_t1_ws.csv", XCAP)
if wL is not None:
    ax.plot(wL, wF, color="#ff7f0e", lw=2.0, ls=":", label="warm-start (RL step 0)")
for i, it in enumerate(ITERS):
    L, F = load(f"token_fve_t1_iter{it}.csv", XCAP)
    if L is None:
        continue
    any_it = True
    ax.plot(L, F, color=CMAP(i / 3 * 0.9), lw=2.2, label=f"RL step {int(it)}")
if any_it:
    if kL is not None:
        ax.plot(kL, kF, color=C_BASE, lw=1.6, ls="--", label=BASE)
    ax.axhline(0, color="k", lw=.6, alpha=.4)
    ax.set_xlim(0, XCAP)
    ax.set_xlabel("explanation truncation length (content tokens)")
    ax.set_ylabel("round-trip FVE")
    ax.set_title("Front-loading emerges over RL training")
    ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{HERE}/fig3_training_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig3 ok")

# ---- fig4: steering first-mention per trait ----
MISS = 10
MODELS = [(OURS, "frontload_v3_t1_raw_judged.json", C_OURS, "D-", 2.3),
          (BASE, "frontload_kitft_t1_raw_judged.json", C_BASE, "s--", 1.8)]
_cache = {}
def curve(fn, trait):
    if fn not in _cache:
        _cache[fn] = json.load(open(os.path.join(HERE, fn)))
    byr = defaultdict(list)
    for x in _cache[fn]:
        if x["trait"] != trait:
            continue
        fi = x.get("first_index")
        if fi == 0:
            continue  # unjudged row (subset judging)
        byr[x["r"]].append(fi if (fi and fi >= 1) else MISS)
    rs = sorted(r for r in byr if len(byr[r]) >= 20)
    return rs, [float(np.mean(byr[r])) for r in rs]

for trait in ("yellow", "sycophancy", "neuroticism"):
    if not all(os.path.exists(os.path.join(HERE, fn)) for _, fn, *_ in MODELS):
        break
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for label, fn, color, style, lw in MODELS:
        rs, mean = curve(fn, trait)
        ax.plot(rs, mean, style, color=color, lw=lw, ms=4.5, alpha=0.9, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("steering strength  r  (log)")
    ax.set_ylabel(f"mean first-mention index of {trait.upper()}   (not present = 10)")
    ax.set_ylim(10.4, 0.6); ax.set_yticks(range(1, 11)); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Where steered {trait.upper()} first appears in the explanation list")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig4_steering_{trait}.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"fig4 {trait} ok")
print("POST_FIGS_DONE")
