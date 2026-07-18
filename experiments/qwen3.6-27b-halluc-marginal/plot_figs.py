"""Figures for the hallucination-marginal comparison (matches evalsuite house style)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED, GREY = "#2a78d6", "#e34948", "#888888"
d = json.load(open(HERE / "results" / "analysis.json"))
S = d["summary"]
mined = d["mined"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# per-case collapse over matched std rollouts
paired = []
for c in mined:
    if not c["std_matches"]:
        continue
    paired.append(dict(
        mat_marg=c["marginal"], mat_solo=c["solo"], mat_loo=c["loo_damage"],
        mat_hall=c["verdict"] in ("CONTRADICTED", "FABRICATED"),
        std_marg=mean([m["marginal"] for m in c["std_matches"]]),
        std_solo=mean([m["solo"] for m in c["std_matches"]]),
        std_loo=mean([m["loo_damage"] for m in c["std_matches"]])))

# ── Fig 1: paired marginal — mat item (always <0) vs matched std sentence ─────
fig, ax = plt.subplots(figsize=(7.6, 6.0))
xm = np.array([p["mat_marg"] for p in paired])
ym = np.array([p["std_marg"] for p in paired])
hall = np.array([p["mat_hall"] for p in paired])
ax.axhline(0, color=GREY, lw=0.9, zorder=1)
ax.axvline(0, color=GREY, lw=0.9, zorder=1)
lim = [-0.9, 1.0]
ax.plot(lim, lim, "--", color=GREY, lw=1, zorder=1, label="y = x")
ax.scatter(xm[~hall], ym[~hall], s=26, c=BLUE, alpha=0.7, edgecolor="none",
           label=f"mat item faithful (n={(~hall).sum()})", zorder=3)
ax.scatter(xm[hall], ym[hall], s=26, c=RED, alpha=0.75, edgecolor="none",
           label=f"mat item hallucinated (n={hall.sum()})", zorder=3)
ax.set_xlim(-0.42, 0.02)
ax.set_ylim(*lim)
ax.set_xlabel("matryoshka item — marginal FVE (mined: always < 0)", fontsize=11)
ax.set_ylabel("matched standard-NLA sentence — marginal FVE", fontsize=11)
ax.set_title("Where the matryoshka critic penalizes an item,\nthe standard critic rewards the same claim",
             fontsize=12.5)
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(color="#ccc", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.03,
         f"n = {len(paired)} mined matryoshka items (top-3, negative marginal) with a matched standard sentence "
         f"(same context, mean over the standard model's matched rollouts).\n"
         f"{(ym > 0).mean()*100:.0f}% of matched standard sentences have POSITIVE marginal (mean "
         f"{ym.mean():+.2f}) — the standard NLA's reconstruction improves on a claim its matryoshka "
         f"counterpart hurts.\nStandard marginals are inflated by its deeply-negative-start cumulative curve; "
         f"see the solo/leave-one-out panel for the order-independent view.",
         fontsize=7.6, color="#777", ha="left", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_paired_marginal.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Fig 2: reconstruction role — marginal / solo / LOO-damage, mined vs matched ─
fig, ax = plt.subplots(figsize=(7.8, 5.2))
metrics = ["marginal FVE", "solo FVE", "LOO damage\n(full − leave-one-out)"]
mat_vals = [np.mean([p["mat_marg"] for p in paired]),
            np.mean([p["mat_solo"] for p in paired]),
            np.mean([p["mat_loo"] for p in paired if p["mat_loo"] is not None])]
std_vals = [np.mean([p["std_marg"] for p in paired]),
            np.mean([p["std_solo"] for p in paired]),
            np.mean([p["std_loo"] for p in paired if p["std_loo"] is not None])]
x = np.arange(3)
w = 0.36
ax.bar(x - w/2, mat_vals, w, color=BLUE, label="matryoshka mined item")
ax.bar(x + w/2, std_vals, w, color=RED, label="matched standard sentence")
ax.axhline(0, color="#444", lw=0.9)
ax.set_ylim(-0.2, 0.5)
for xi, (mv, sv) in enumerate(zip(mat_vals, std_vals)):
    ax.text(xi - w/2, mv - 0.012, f"{mv:+.3f}", ha="center",
            fontsize=9, color=BLUE, va="top")
    if sv > 0.35:   # tall bar: label inside so it clears the legend/title
        ax.text(xi + w/2, sv - 0.02, f"{sv:+.3f}", ha="center",
                fontsize=9, color="white", va="top", fontweight="bold")
    else:
        ax.text(xi + w/2, sv + 0.012, f"{sv:+.3f}", ha="center",
                fontsize=9, color=RED, va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.tick_params(axis="x", pad=8)
ax.set_ylabel("mean value over paired cases", fontsize=11)
ax.set_title("Same claim, opposite reconstruction role\n"
             "matryoshka: redundant/harmful · standard: load-bearing", fontsize=12.5)
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", color="#ccc", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.02,
         "Solo FVE = the item scored alone. LOO damage > 0 means removing the item HURTS reconstruction "
         "(load-bearing); ≈0 or < 0 means redundant/harmful.\nThe matched standard sentence reconstructs "
         "well alone (+0.25) and is load-bearing (+0.09); the matryoshka item does neither.",
         fontsize=7.8, color="#777", ha="left", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_reconstruction_role.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Fig 3: is negative marginal a hallucination detector? P(halluc | sign) ────
fig, ax = plt.subplots(figsize=(7.4, 5.0))
models = [("matryoshka", "mat", BLUE), ("standard", "std", RED)]
x = np.arange(2)
w = 0.36
neg = [S[m]["p_halluc_given_neg"] for _, m, _ in models]
pos = [S[m]["p_halluc_given_pos"] for _, m, _ in models]
cols = [c for *_, c in models]
ax.bar(x - w/2, neg, w, color=cols)
ax.bar(x + w/2, pos, w, color=cols, alpha=0.4)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#555", label="P(hallucinated | marginal < 0)"),
                   Patch(facecolor="#555", alpha=0.4, label="P(hallucinated | marginal ≥ 0)")],
          fontsize=9.5, loc="upper right")
for xi, (n, p) in enumerate(zip(neg, pos)):
    ax.text(xi - w/2, n + 0.008, f"{n:.2f}", ha="center", fontsize=9.5)
    ax.text(xi + w/2, p + 0.008, f"{p:.2f}", ha="center", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels([nm for nm, *_ in models], fontsize=11)
ax.set_ylabel("P(item judged hallucinated)", fontsize=11)
ax.set_ylim(0, 0.72)
ax.set_title("Negative marginal FVE is a weak hallucination signal\n"
             "(solid = negative-marginal items, faded = the rest)", fontsize=12.5)
ax.grid(axis="y", color="#ccc", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.02,
         f"Hallucinated = judged CONTRADICTED or FABRICATED by {S['judge']}. "
         f"Matryoshka: negative marginal barely lifts hallucination rate ({neg[0]:.2f} vs {pos[0]:.2f}); "
         f"standard: a modest, real lift ({neg[1]:.2f} vs {pos[1]:.2f}).\nNeither critic's per-item FVE sign "
         f"is a clean hallucination detector — most hallucinated items still carry positive marginal FVE.",
         fontsize=7.8, color="#777", ha="left", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_halluc_detector.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved 3 figures to results/")
