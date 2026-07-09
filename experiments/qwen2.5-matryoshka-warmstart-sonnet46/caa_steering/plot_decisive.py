"""Figures for the corrected (decisive) result, run locally on CPU.
fig_contrast_verbalize.png : LLM-judged verbalization vs strength, per trait, neutral-negative
                             (retains surface) vs opposite-stance negative (cancels surface).
fig_steering_pmatch.png    : p(behavior-matching) vs steering multiplier — in-distribution prose
                             vectors (flat) vs the OOD answer-letter corrigibility vector (steers).
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
TCOL = {"corrigibility": "#d62728", "sycophancy": "#1f77b4", "sadness": "#9467bd", "yellow": "#e8b800"}


# ---- fig 1: verbalization by contrast type ----
rows = list(csv.DictReader(open(os.path.join(R, "verbalize_judged_agg.csv"))))
piv = {}
for r in rows:
    piv.setdefault(r["trait"], {})[float(r["strength"])] = float(r["judge_appearance_rate"])
groups = {"corrigibility": ("corr_neu", "corr_opp"), "sycophancy": ("syc_neu", "syc_opp"),
          "sadness": ("sad_neu", "sad_opp"), "yellow": ("yel_neu", "yel_opp")}
fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
for ax, (trait, (neu, opp)) in zip(axes, groups.items()):
    for name, ls, lab in [(neu, "-", "neutral neg (retains surface)"), (opp, "--", "opposite-stance neg (cancels surface)")]:
        d = piv.get(name, {})
        if d:
            xs = sorted(d); ax.plot(xs, [d[x] for x in xs], marker="o", ls=ls, lw=2, color=TCOL[trait], label=lab)
    ax.set_title(trait); ax.set_xlabel("steering strength r"); ax.grid(alpha=.3); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7.5, loc="upper right")
axes[0].set_ylabel("LLM-judged verbalization rate")
fig.suptitle("The AV verbalizes abstract dispositions too — when the contrast keeps their surface.\n"
             "Same trait, two CAA contrasts: neutral negative (retains content) vs opposite-stance negative (cancels it).", y=1.10, fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_contrast_verbalize.png"), dpi=130, bbox_inches="tight")
print("wrote fig_contrast_verbalize.png")


# ---- fig 2: steering p(matching) ----
def load_pm(fn):
    out = {}
    for r in csv.DictReader(open(os.path.join(R, fn))):
        k = r.get("vector") or r.get("trait")
        xs = [(float(c[1:]), float(v)) for c, v in r.items() if c.startswith("m")]
        out[k] = sorted(xs)
    return out


indist = load_pm("decisive_steer_pmatch.csv")      # in-distribution prose vectors
ablet = load_pm("steer_pmatch.csv")                # OOD answer-letter vectors (earlier run)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
# the one vector that steers: answer-letter corrigibility
if "corrigibility" in ablet:
    x = [a for a, _ in ablet["corrigibility"]]; y = [b for _, b in ablet["corrigibility"]]
    ax.plot(x, y, marker="s", lw=2.5, color="#d62728", label="corrigibility — OOD answer-letter vector (STEERS)")
for name, col, lab in [("corr_neu", "#d62728", "corrigibility — in-dist prose vector"),
                       ("sad_neu", "#9467bd", "sadness — in-dist prose vector"),
                       ("syc_neu", "#1f77b4", "sycophancy — in-dist prose vector"),
                       ("yel_neu", "#e8b800", "yellow — in-dist prose vector")]:
    if name in indist:
        x = [a for a, _ in indist[name]]; y = [b for _, b in indist[name]]
        ax.plot(x, y, marker="o", ls=":", lw=1.8, color=col, alpha=0.85, label=lab)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("steering multiplier (× unit v̂ added to layer-20 residual, all positions)")
ax.set_ylabel("p(behavior-matching answer), held-out A/B")
ax.set_title("Only the OOD answer-letter corrigibility vector actually steers behavior;\nthe in-distribution prose vectors (which the AV reads) are flat")
ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="center left")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_steering_pmatch.png"), dpi=130, bbox_inches="tight")
print("wrote fig_steering_pmatch.png")
