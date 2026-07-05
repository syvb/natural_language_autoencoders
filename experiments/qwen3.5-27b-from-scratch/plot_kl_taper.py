"""Figure: the position-tapered KL coefficient vs the flat v3 baseline.

coef(t) = KL_LOSS_COEF * max(0.5 ** (t / NLA_KL_TAPER_HALF_LIFE), FLOOR)
with the production values (KL_LOSS_COEF=0.02, half-life 40 tokens, floor 0).
Writes kl_taper_curve.png next to this script.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COEF = 0.02
HALF_LIFE = 40.0
T_MAX = 160          # --rollout-max-response-len
BUDGET_MAX = 120     # truncation budget U[1, 120] — where the reward reaches

t = np.arange(0, T_MAX + 1)
taper = COEF * 0.5 ** (t / HALF_LIFE)

fig, ax = plt.subplots(figsize=(7.6, 4.4))

# Reward reach: the truncation budget region (context, not a series).
ax.axvspan(0, BUDGET_MAX, color="#9467bd", alpha=0.05, lw=0)
ax.text(BUDGET_MAX / 2, 0.0212, "truncation budget  U[1, 120]  (where the reward reaches)",
        ha="center", fontsize=8.5, color="#666666")
ax.axvline(BUDGET_MAX, color="#bbbbbb", lw=0.8, ls=":")

# Flat baseline (v3 recipe): dashed neutral reference.
ax.axhline(COEF, color="#444444", lw=2, ls="--")
ax.text(T_MAX, COEF + 0.0004, "flat KL 0.02 (7B v3 recipe)",
        ha="right", fontsize=9, color="#444444")

# The taper.
ax.plot(t, taper, color="#9467bd", lw=2.4)
ax.text(24, 0.0160, "tapered KL\ncoef(t) = 0.02 · 0.5^(t/40)",
        fontsize=9.5, color="#9467bd")

# Half-life markers with direct value labels.
for k in (1, 2, 3):
    x = k * HALF_LIFE
    y = COEF * 0.5 ** k
    ax.plot([x], [y], "o", ms=6, color="#9467bd", mfc="white", mew=1.6)
    ax.annotate(f"{y:.4f}", (x, y), xytext=(6, 5), textcoords="offset points",
                fontsize=8.5, color="#555555")

# Measured batch-mean from the pipeline smoke (real rollouts, U[1,120] draws).
ax.axhline(0.0131, color="#9467bd", lw=1, ls=":", alpha=0.6)
ax.text(T_MAX, 0.0134, "batch-mean kl_coef_eff ≈ 0.013 (measured, smoke)",
        ha="right", fontsize=8, color="#8a6db1")

ax.set_xlim(0, T_MAX)
ax.set_ylim(0, 0.022)
ax.set_xlabel("response token position  t")
ax.set_ylabel("KL coefficient applied at t")
ax.set_title("Position-tapered KL: full strength at the front, halving every 40 tokens")
ax.grid(alpha=0.25, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kl_taper_curve.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
