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
FAITH = json.load(open(HERE / "results" / "judged_faithfulness.json"))
HALL = {"CONTRADICTED", "FABRICATED"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def phalluc_by_stratum(arm, fname, kmax):
    """(k, marginal, is_halluc) rows, then P(halluc|neg)/P(halluc|pos) for
    pooled / first-item(k=0) / later-items(k>=1). Recomputed from raw files so
    the figure never trusts a precomputed pooled number."""
    rows = []
    for e in json.load(open(HERE / "results" / fname))["entries"]:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            m = _marg(rec)
            for k in range(min(kmax, len(m))):
                v = FAITH.get(f"{arm}|{e['ci']}|{ri}|{k}")
                if v is not None:
                    rows.append((k, m[k], v in HALL))
    out = {}
    for name, sub in [("pooled", rows),
                      ("k=0", [r for r in rows if r[0] == 0]),
                      ("k≥1", [r for r in rows if r[0] >= 1])]:
        neg = [h for _, mg, h in sub if mg < 0]
        pos = [h for _, mg, h in sub if mg >= 0]
        out[name] = (float(np.mean(neg)) if neg else float("nan"), len(neg),
                     float(np.mean(pos)) if pos else float("nan"), len(pos))
    return out


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
YTOP = 1.6
ax.plot([-0.9, YTOP], [-0.9, YTOP], "--", color=GREY, lw=1, zorder=1, label="y = x")
# clip-marker: points above the axis top are drawn as up-triangles at the top
above = ym > YTOP
ax.scatter(xm[~hall & ~above], ym[~hall & ~above], s=26, c=BLUE, alpha=0.7,
           edgecolor="none", label=f"mat item faithful (n={(~hall).sum()})", zorder=3)
ax.scatter(xm[hall & ~above], ym[hall & ~above], s=26, c=RED, alpha=0.75,
           edgecolor="none", label=f"mat item hallucinated (n={hall.sum()})", zorder=3)
ax.scatter(xm[above], np.full(above.sum(), YTOP - 0.02), s=30, marker="^",
           c=[RED if h else BLUE for h in hall[above]], alpha=0.8, edgecolor="none", zorder=3)
ax.set_xlim(-0.42, 0.02)
ax.set_ylim(-0.9, YTOP)
ax.set_xlabel("matryoshka item — marginal FVE (mined: always < 0)", fontsize=11)
ax.set_ylabel("matched standard-NLA sentence — marginal FVE", fontsize=11)
ax.set_title("Where the matryoshka critic penalizes an item,\nthe standard critic rewards the same claim",
             fontsize=12.5)
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(color="#ccc", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, -0.03,
         f"n = {len(paired)} mined matryoshka items (top-3, negative marginal) with a matched standard sentence "
         f"(same context, mean over the standard model's matched rollouts); "
         f"{int(above.sum())} points with std marginal > {YTOP} shown as ▲ at the top edge.\n"
         f"{(ym > 0).mean()*100:.0f}% of matched standard sentences have POSITIVE marginal (mean "
         f"{ym.mean():+.2f}) — the standard NLA's reconstruction improves on a claim its matryoshka "
         f"counterpart hurts.\nBUT standard marginals are confounded by its deeply-negative-start cumulative "
         f"curve (a second sentence climbs out of a −0.6 hole); the order-independent solo/leave-one-out "
         f"panel is the clean comparison.",
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

# ── Fig 3: negative marginal is NOT a detector — std "lift" is a position artifact ─
from matplotlib.patches import Patch
mat_str = phalluc_by_stratum("mat", "subset_scores_mat.json", 3)
std_str = phalluc_by_stratum("std", "subset_scores_std.json", 99)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), sharey=True)
strata = ["pooled", "k=0", "k≥1"]
strata_lbl = ["pooled\n(all items)", "first item\n(k=0)", "later items\n(k≥1)"]
x = np.arange(3)
w = 0.38
for ax, (name, col, data) in zip(
        axes, [("matryoshka", BLUE, mat_str), ("standard", RED, std_str)]):
    neg = [data[s][0] for s in strata]
    pos = [data[s][2] for s in strata]
    ax.bar(x - w/2, neg, w, color=col, label="marginal < 0")
    ax.bar(x + w/2, pos, w, color=col, alpha=0.38, label="marginal ≥ 0")
    for xi, s in enumerate(strata):
        pn, nn, pp, npp = data[s]
        ax.text(xi - w/2, pn + 0.01, f"{pn:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + w/2, pp + 0.01, f"{pp:.2f}", ha="center", fontsize=8.5)
        ax.text(xi - w/2, 0.03, f"n={nn}", ha="center", fontsize=7, color="white", rotation=90, va="bottom")
        ax.text(xi + w/2, 0.03, f"n={npp}", ha="center", fontsize=7,
                color=col if pp < 0.15 else "white", rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(strata_lbl, fontsize=9.5)
    ax.set_title(name, fontsize=12, color=col, fontweight="bold")
    ax.grid(axis="y", color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("P(item judged hallucinated)", fontsize=11)
axes[0].set_ylim(0, 0.82)
axes[0].legend(handles=[Patch(facecolor="#555", label="P(halluc | marginal < 0)"),
                        Patch(facecolor="#555", alpha=0.38, label="P(halluc | marginal ≥ 0)")],
               fontsize=9, loc="upper right")
fig.suptitle("Negative marginal FVE is not a hallucination detector — the standard model's\n"
             "pooled “signal” is a position artifact that reverses within later items",
             fontsize=13, y=1.02)
fig.text(0.01, -0.04,
         f"Hallucinated = CONTRADICTED or FABRICATED ({S['judge']}). The standard critic scores a LONE first "
         f"sentence at ≈−0.58 FVE (94% negative) — so most std “negative marginals” are just first items, "
         f"and P(halluc) falls with position for positional reasons.\nStratifying removes the confound: "
         f"standard pooled 0.58>0.52 looks like a signal (Fisher p=0.001), but within later items it REVERSES "
         f"to 0.37<0.52 (OR 0.54, p=3e-9) — a genuine mid-explanation negative marginal is if anything LESS "
         f"often a hallucination.\nMatryoshka shows no lift at any position (ORs ≈1.0, all n.s.). Conclusion: "
         f"neither critic's per-item FVE sign detects hallucination.",
         fontsize=7.6, color="#777", ha="left", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_halluc_detector.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved 3 figures to results/")
