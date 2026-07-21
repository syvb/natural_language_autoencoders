"""Head-to-head mat-vs-std comparison figure for the rhyme-steering probe.

One metric — steering success (second line rhymes with the edit target "moon",
out of 10 seed-matched T=1 samples) — across matched edit granularities,
ordered by edit scope. Reads results/rhyme_scores.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MAT, STD = "#2a78d6", "#e34948"          # entity colors, fixed across this series
s = json.load(open(HERE / "results" / "rhyme_scores.json"))

GROUPS = [  # (label, mat key, std key)
    ("no edit\n(round-trip)", "mat:orig", "std:orig"),
    ("abstract rhyme claim\n“rhymes with bed”→“June”", "mat:edit_rhyme_moon", "std:edit_rhyme_moon"),
    ("single most-direct bit\nmat: leading item\nstd: ‘final token’ clause", "mat:edit_lead_moon", "std:edit_finaltok_moon"),
    ("quoted line-1 text\n(all quote mentions)", "mat:edit_quotes_moon", "std:edit_quotes_moon"),
    ("full edit\n(every reference)", "mat:edit_full_moon", "std:edit_full_moon"),
]
CEIL = s["donor_moon"]["M"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(figsize=(9.4, 5.4))
xs = np.arange(len(GROUPS))
W, OFF = 0.32, 0.18
for off, col, name, ki in [(-OFF, MAT, "matryoshka", 1), (OFF, STD, "standard", 2)]:
    for x, g in zip(xs, GROUPS):
        k, n = s[g[ki]]["M"], s[g[ki]]["n"]
        lo, hi = wilson(k, n)
        ax.bar(x + off, max(k, 0.09), width=W, color=col, zorder=3)   # stub keeps zero bars identifiable
        ax.errorbar(x + off, k, yerr=[[k - lo * n], [hi * n - k]], fmt="none",
                    ecolor="#555", elinewidth=1.1, capsize=3, zorder=4)
        ax.text(x + off, max(k, hi * n) + 0.28, f"{k}", ha="center", fontsize=9.5,
                color="#333", fontweight="bold")
ax.axhline(CEIL, color="#666", lw=1.2, ls=(0, (5, 3)), zorder=2)
ax.text(-0.44, CEIL + 0.15, f"ceiling: real “moon” activation patched in ({CEIL}/10)",
        ha="left", fontsize=8.4, color="#555")
ax.set_xticks(xs)
ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8.8)
ax.set_ylim(0, 10.6)
ax.set_yticks(range(0, 11, 2))
ax.set_ylabel("steered continuations /10\n(second line rhymes with “moon”)", fontsize=10)
ax.set_xlabel("what was edited in the explanation  (increasing edit scope →)", fontsize=10)
ax.set_title("Steering the couplet by editing the NLA explanation:\n"
             "matryoshka steers fully from ONE bullet — standard only once its quoted text is rewritten",
             fontsize=11.5)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MAT, label="matryoshka (lines)"),
                   plt.Rectangle((0, 0), 1, 1, color=STD, label="standard (sentences)")],
          fontsize=9.5, loc="upper left")
ax.grid(axis="y", color="#ccc", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.02, -0.015,
         "Each explanation variant is re-encoded by the arm's own critic and patched into the L42 residual at the final token of\n"
         "“The sun goes down to rest its head”; n=10 T=1 samples per bar, seed-matched. Whiskers: Wilson 95% CI. Unpatched\n"
         "baseline and self-patch controls: 0/10 moon (9/10 head-rhyme). Bars at 0 mean the un-edited rhyme survived intact.",
         fontsize=7.6, color="#777", ha="left", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_mat_vs_std_steer.png", dpi=150, bbox_inches="tight")
print("[saved] results/fig_mat_vs_std_steer.png")
