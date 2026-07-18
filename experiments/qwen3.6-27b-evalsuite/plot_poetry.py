"""Figures for the Planning-in-Poetry reproduction (poetry_steer.py).

fig_poetry_ladder.png      steered-to-target rate per intervention arm, one
                           panel per couplet: single-token arms are all zero;
                           whole-line (11-token) arms work, including the
                           paper's explanation-edit protocol on the spont
                           couplet.
fig_poetry_planmention.png does the explanation at the line-break token name
                           the upcoming rhyme word? (n=30 samples/model)

Usage: python plot_poetry.py <dir with po_tally.json + po_av_*.json>
"""
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results"
T = json.load(open(f"{D}/po_tally.json"))

C = {"mat": "#2a78d6", "std": "#eb6834"}
GRAY = "#898781"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
NAME = {"mat": "matryoshka", "std": "standard"}

# ── ladder ───────────────────────────────────────────────────────────────────
# rows: (label, kind) — kind "single" = one gray bar from arm key,
#                      kind "pair"   = mat/std bars from key template
ROWS = [
    ("no patch", "single", "nopatch"),
    ("edited NLA vec, line-break token only\n(best over both NLAs, α ≤ 4)", "single", None),
    ("true activation swap, line-break\ntoken only (best of 10 layers)", "single", None),
    ("true activations, all 11 tokens, L42", "single", "xactL42_all"),
    ("true activations, all 11 tokens,\nbest layer", "single", None),
    ("NLA recons of the OTHER couplet's\nexplanations, all 11 tokens, L42", "pair", "mrec_{m}_xorig_all"),
    ("NLA recons of the EDITED\nexplanations, all 11 tokens, L42", "pair", "mrec_{m}_edit_all"),
]
BEST_SINGLE_NOTE = {"paper": "L24", "spont": "L12"}


def best_over(tag, pat):
    vals = [v["target"] for k, v in T[tag].items() if re.fullmatch(pat, k)]
    return max(vals) if vals else 0.0


fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharex=True, facecolor=SURF)
for ax, tag, title in zip(
        axes, ["paper", "spont"],
        ["paper couplet (carrot → grab it → “rabbit”)\nedit: rabbit→mouse — prompt names the rabbit",
         "spontaneous couplet (cat → mouse → “house”)\nedit: house→habit — no prompt hint"]):
    ax.set_facecolor(SURF)
    y = 0
    yticks, ylabels = [], []
    for label, kind, key in ROWS[::-1]:
        if kind == "single":
            if key is None and "α" in label:
                v = best_over(tag, r"(mat|std)_(steer|patch_edit)_(full|k1|k2).*")
            elif key is None and "10 layers" in label:
                v = max(best_over(tag, r"xactL\d+_patch"), T[tag]["xact_patch"]["target"])
            elif key is None:
                v = best_over(tag, r"xactL\d+_all")
            else:
                v = T[tag][key]["target"]
            ax.barh(y, v, height=0.55, color=GRAY, zorder=3)
            note = f" ({BEST_SINGLE_NOTE[tag]})" if (key is None and "best layer" in label) else ""
            ax.annotate(f"{v:.0%}{note}", (v, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK2)
            yticks.append(y); ylabels.append(label)
            y += 1.0
        else:
            for j, m in enumerate(("std", "mat")):
                arm = key.format(m=m)
                v = T[tag][arm]["target"]
                lo = T[tag][arm].get("loose", v)
                yy = y + j * 0.42
                if lo > v:
                    ax.barh(yy, lo, height=0.36, color=C[m], alpha=0.35, zorder=2)
                ax.barh(yy, v, height=0.36, color=C[m], zorder=3)
                txt = f"{v:.0%}" + (f" ({lo:.0%} incl. “habitat”)" if lo > v else "")
                ax.annotate(f"{NAME[m]} {txt}", (max(v, lo), yy), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8.5,
                            color=INK)
            yticks.append(y + 0.21); ylabels.append(label)
            y += 1.55
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("second lines ending in the steered-to rhyme family", color=INK)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
axes[1].set_yticklabels([])
fig.suptitle("Rewriting the rhyme plan: single-token steering is causally inert on Qwen3.6-27B; "
             "the plan is diffuse across the line",
             x=0.02, ha="left", fontsize=12, color=INK)
fig.text(0.02, 0.012,
         "n=25 seed-matched T=1 completions per arm · patch = norm-matched replacement at the layer's "
         "block output during prefill · NLA arms: each model's own critic recons of per-token explanations",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.035, 1, 0.93))
fig.savefig(f"{D}/fig_poetry_ladder.png", dpi=170, facecolor=SURF)
print("saved fig_poetry_ladder.png")

# ── plan mention at the line break ───────────────────────────────────────────
PLAN = {"paper": r"\brabbit", "spont": r"\bhouse"}
rates = {}
for m in ("mat", "std"):
    av = json.load(open(f"{D}/po_av_{m}.json"))["tags"]
    for tag in ("paper", "spont"):
        gens = av[tag]["gens"]
        last = str(max(int(k) for k in gens))
        g = gens[last]
        rates[(m, tag)] = (sum(bool(re.search(PLAN[tag], x, re.I)) for x in g), len(g))

fig2, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURF)
ax.set_facecolor(SURF)
groups = [("spont", "“house” — the planned rhyme\n(never appears in prompt or text)"),
          ("paper", "“rabbit” — planned rhyme, but\nalso named in the user prompt")]
for gi, (tag, glabel) in enumerate(groups):
    for j, m in enumerate(("mat", "std")):
        k, n = rates[(m, tag)]
        x = gi * 1.2 + (j - 0.5) * 0.38
        ax.bar(x, k / n, width=0.34, color=C[m], zorder=3)
        ax.annotate(f"{k}/{n}", (x, k / n), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
        if gi == 0:
            ax.annotate(NAME[m], (x, k / n), xytext=(0, 18),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=C[m], fontweight="bold")
ax.set_xticks([0, 1.2])
ax.set_xticklabels([g[1] for g in groups], fontsize=9, color=INK)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_title("Does the explanation at the line-break token name the upcoming rhyme word?",
             fontsize=11, loc="left", color=INK)
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
ax.tick_params(colors=INK2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
fig2.text(0.02, 0.015, "explanations sampled at T=1 at the newline ending the couplet's first line",
          color=INK2, fontsize=8)
fig2.tight_layout(rect=(0, 0.04, 1, 1))
fig2.savefig(f"{D}/fig_poetry_planmention.png", dpi=170, facecolor=SURF)
print("saved fig_poetry_planmention.png")
