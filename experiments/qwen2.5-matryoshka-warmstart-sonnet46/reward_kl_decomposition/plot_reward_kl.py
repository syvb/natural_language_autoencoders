"""Figures for the v3 reward/KL per-token-position decomposition. Local,
matplotlib only — reads aggregate.csv (+ per_sample.npz for the length
histogram) produced by sweep_reward_kl.py.

Both series are in the same units (reward per token at position t):
  reconstruction = marginal reward dr(t) = r(t) - r(t-1)
  KL penalty     = beta * k1(t)   (beta = 0.03, the v3 kl-loss-coef)
dr(1) is excluded from the marginal panels: r(0) is the failed-extraction
floor (-2.0), so dr(1) ~ +1.9 is a tag-vs-floor artifact, not reconstruction.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("RESULTS", os.path.join(HERE, "results"))
BETA = float(os.environ.get("KL_COEF", "0.03"))
XCAP = int(os.environ.get("XCAP", "120"))  # trained truncation range U[1,120]
WIN = int(os.environ.get("WIN", "5"))      # rolling-mean window (tokens)

C_REC = "#1f77b4"   # reconstruction (marginal reward)
C_KL = "#d62728"    # KL penalty (k1, the trained estimator)
C_KLX = "#ff9896"   # exact full-vocab KL (low-variance check)

rows = list(csv.DictReader(open(os.path.join(R, "aggregate.csv"))))
t = np.array([int(r["t"]) for r in rows])
r_t = np.array([float(r["mean_reward"]) for r in rows])
dr = np.array([float(r["mean_marginal"]) for r in rows])
k1 = np.array([float(r["mean_k1"]) for r in rows])
klx = np.array([float(r["mean_kl_exact"]) for r in rows])
expo = np.array([float(r["exposure"]) for r in rows])
n_alive = np.array([int(r["n_alive"]) for r in rows])


def roll(y, w=WIN):
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


m = t <= XCAP
m2 = m & (t >= 2)  # marginal panels: drop the t=1 floor artifact

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(t[m2], roll(dr[m2]), color=C_REC, lw=2.0, label="reconstruction: marginal reward Δr(t)")
a1.plot(t[m], roll(BETA * k1[m]), color=C_KL, lw=2.0, label=f"KL penalty: β·k1(t), β={BETA}")
a1.plot(t[m], roll(BETA * klx[m]), color=C_KLX, lw=1.4, ls="--", label="β·KL_exact(t) (check)")
a1.axhline(0, color="gray", lw=0.8)
a1.set_xlabel("token position t in the rollout")
a1.set_ylabel("reward units per token")
a1.set_title(f"raw per-position signal ({WIN}-token rolling mean)")
a1.grid(alpha=.3)
a1.legend(fontsize=9)

a2.plot(t[m2], roll((dr * expo)[m2]), color=C_REC, lw=2.0, label="Δr(t)·P(L≥t)")
a2.plot(t[m], roll((BETA * k1 * expo)[m]), color=C_KL, lw=2.0, label="β·k1(t)·P(L≥t)")
a2.axhline(0, color="gray", lw=0.8)
a2.set_xlabel("token position t in the rollout")
a2.set_ylabel("expected reward units per token")
a2.set_title("exposure-weighted (as trained, L ~ U[1,120])")
a2.grid(alpha=.3)
a2.legend(fontsize=9)
fig.suptitle("v3 matryoshka RL (iter_0000200): reconstruction vs KL contribution per token position", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_reward_kl_pertoken.png"), dpi=130, bbox_inches="tight")

fig2, (b1, b2) = plt.subplots(1, 2, figsize=(13, 5))
b1.plot(t[m], r_t[m], color=C_REC, lw=2.0, label="reward r(t) of the first-t-token prefix")
b1.axhline(0, color="gray", lw=0.8)
b1.set_xlabel("prefix length t (content tokens)")
b1.set_ylabel("reward = −mse_nrm")
b1.set_title("prefix reward curve")
b1.grid(alpha=.3)
b1.legend(fontsize=9)

cum_rec = np.cumsum(np.where(t >= 2, dr, 0.0))  # reward gained after token 1
cum_kl = BETA * np.cumsum(k1)
b2.plot(t[m], cum_rec[m], color=C_REC, lw=2.0, label="Σ Δr (reward gained since t=1)")
b2.plot(t[m], cum_kl[m], color=C_KL, lw=2.0, label="Σ β·k1 (total KL penalty paid)")
b2.plot(t[m], (cum_rec - cum_kl)[m], color="k", lw=1.4, ls=":", label="net")
b2.axhline(0, color="gray", lw=0.8)
b2.set_xlabel("rollout length t")
b2.set_ylabel("cumulative reward units")
b2.set_title("cumulative decomposition")
b2.grid(alpha=.3)
b2.legend(fontsize=9)
fig2.suptitle("v3 matryoshka RL: reward level and cumulative reconstruction-vs-KL budget", y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(R, "fig_reward_kl_cumulative.png"), dpi=130, bbox_inches="tight")

# headline numbers
xover = next((int(tt) for tt, d, k in zip(t[m2], roll(dr[m2]), roll(BETA * k1[m2]))
              if d < k), None)
print(f"r(1)={r_t[0]:.3f}  r({XCAP})={r_t[t <= XCAP][-1]:.3f}")
print(f"mean beta*k1 over t<=({XCAP}): {BETA * k1[m].mean():.5f} /token")
print(f"first position where smoothed beta*k1 exceeds marginal reconstruction: {xover}")
print(f"n_alive at t={XCAP}: {n_alive[m][-1]}")
