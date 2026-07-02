"""Figure for the telegraphic-code investigation: variant bars (v3rl critic) +
cross-critic heatmap. Reads v3_faithfulness_results/code_invest_matrix.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "v3_faithfulness_results")
m = json.load(open(f"{D}/code_invest_matrix.json"))["matrix"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

BARS = [("orig", "original (telegraphic, quotes)", "#9467bd"),
        ("para_keepquotes", "paraphrase EXCEPT quotes", "#2ca02c"),
        ("fluent", "fluent rewrite (quotes kept)", "#2ca02c"),
        ("wordshuffle", "words shuffled within lines", "#8c8c8c"),
        ("quotesonly", "quoted spans ONLY", "#1f77b4"),
        ("para", "full paraphrase (quotes lost)", "#d62728"),
        ("noquote", "quotes stripped", "#d62728")]
ys = [m["v3rl"][k] for k, _, _ in BARS]
labels = [l for _, l, _ in BARS]
cols = [c for _, _, c in BARS]
a1.barh(range(len(BARS))[::-1], ys, color=cols, alpha=.85)
for i, y in enumerate(ys):
    a1.text(max(y, 0) + 0.012, len(BARS) - 1 - i, f"{y:.3f}", va="center", fontsize=10)
a1.set_yticks(range(len(BARS))[::-1]); a1.set_yticklabels(labels, fontsize=10)
a1.axvline(0, color="k", lw=.8)
a1.set_xlabel("round-trip FVE (v3 RL critic, 100 held-out)")
a1.set_title("What carries the signal? Keep the quotes → FVE survives.\nLose the quotes → FVE collapses. Style is irrelevant.")
a1.set_xlim(-0.05, 0.75); a1.grid(alpha=.3, axis="x")

VAR = ["orig", "para_keepquotes", "fluent", "wordshuffle", "quotesonly", "para",
       "noquote", "ws_orig", "ws_para", "gold", "kitft_orig"]
CR = ["v3rl", "v3ws", "kitft"]
M = np.array([[m[c][v] for c in CR] for v in VAR])
im = a2.imshow(M, cmap="RdYlGn", vmin=-0.7, vmax=0.75, aspect="auto")
a2.set_xticks(range(len(CR))); a2.set_xticklabels(["v3 RL critic", "v3 warm-start critic", "kitft critic"])
a2.set_yticks(range(len(VAR))); a2.set_yticklabels(VAR, fontsize=9)
for i in range(len(VAR)):
    for j in range(len(CR)):
        a2.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="black")
a2.set_title("Cross-critic FVE matrix\n(v3 text reads ~equally under the NON-co-trained ws critic)")
fig.colorbar(im, ax=a2, shrink=.8, label="FVE")
fig.suptitle("Is the v3 AV using a telegraphic code? No — the channel is QUOTED TEXT FRAGMENTS", y=1.03, fontsize=13)
fig.tight_layout()
out = f"{D}/fig_code_investigation.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
