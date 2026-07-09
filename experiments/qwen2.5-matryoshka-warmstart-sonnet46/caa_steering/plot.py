"""Plots from the LLM-judged CAA results (run locally, CPU).
fig_methods.png  : judge appearance-rate vs strength, 3 vector-construction methods x 4 traits.
fig_strength.png : magnitude sweep (base + c*||base||*vhat), judge appearance-rate + rank vs c.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
TRAITS = ["sycophancy", "corrigibility", "refusal", "yellow"]
COL = {"sycophancy": "#1f77b4", "corrigibility": "#d62728", "refusal": "#2ca02c", "yellow": "#e8b800"}


def load(fn):
    rows = list(csv.DictReader(open(os.path.join(R, fn))))
    d = {}
    for r in rows:
        d.setdefault(r["trait"], []).append((float(r["strength"]), float(r["judge_appearance_rate"]),
                                             float(r["judge_mean_rank"]) if r["judge_mean_rank"] not in ("", "nan") else np.nan))
    for t in d: d[t].sort()
    return d


methods = [("rawpairs_judged_agg.csv", "raw hand-written pairs", "o", "-"),
           ("caa_ab_judged_agg.csv", "A/B answer-letter (paper)", "s", "--"),
           ("caa_abtext_judged_agg.csv", "A/B full answer-text", "^", ":")]
data = {lbl: load(fn) for fn, lbl, _, _ in methods}

fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for ax, tr in zip(axes, TRAITS):
    for (fn, lbl, mk, ls) in methods:
        xy = data[lbl].get(tr, [])
        if xy:
            x = [a for a, _, _ in xy]; y = [b for _, b, _ in xy]
            ax.plot(x, y, marker=mk, ls=ls, lw=1.8, color=COL[tr], alpha=0.9, label=lbl)
    ax.set_title(tr); ax.set_xlabel("steering strength r"); ax.grid(alpha=.3); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("LLM-judged appearance rate")
fig.suptitle("Does the AV verbalize a CAA-steered trait? (LLM-judged, not regex) — by vector-construction method", y=1.03, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_methods.png"), dpi=130, bbox_inches="tight"); print("wrote fig_methods.png")

# strength (magnitude) sweep
st = load("strength_judged_agg.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for tr in TRAITS:
    xy = st.get(tr, [])
    if not xy: continue
    x = np.array([a for a, _, _ in xy]); ar = np.array([b for _, b, _ in xy]); rk = np.array([c for _, _, c in xy])
    xp = np.where(x == 0, 0.12, x)  # show c=0 on log axis
    a1.plot(xp, ar, marker="o", lw=2, color=COL[tr], label=tr)
    if np.isfinite(rk).any(): a2.plot(xp, rk, marker="o", lw=2, color=COL[tr], label=tr)
a1.set_xscale("log"); a1.set_xlabel("CAA multiplier c   (inj norm ≈ c·128; AV trained at 150)"); a1.set_ylabel("LLM-judged appearance rate")
a1.set_title("Appearance vs steering magnitude\n(base + c·‖base‖·v̂, no renormalization)"); a1.grid(alpha=.3, which="both"); a1.legend(); a1.set_ylim(-0.03, 1.03)
a2.set_xscale("log"); a2.set_xlabel("CAA multiplier c"); a2.set_ylabel("judged mean rank (1=top)")
a2.set_title("When it appears, how high in the list?"); a2.grid(alpha=.3, which="both"); a2.legend(); a2.invert_yaxis()
fig.suptitle("Paper-faithful (answer-letter) CAA: cranking magnitude — LLM-judged", y=1.02, fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_strength.png"), dpi=130, bbox_inches="tight"); print("wrote fig_strength.png")
