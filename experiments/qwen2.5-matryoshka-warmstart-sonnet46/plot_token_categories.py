"""Forest plot: index-controlled excess marginal ΔFVE per token category,
v3 RL vs v3 warm-start. Reads v3_faithfulness_results/token_category_analysis.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "v3_faithfulness_results")
res = json.load(open(f"{D}/token_category_analysis.json"))

ORDER = ["after_quote_tok", "subword_cont", "stopword", "line_last", "capitalized",
         "numeric", "schema_word", "punct_only", "in_quote", "newline",
         "quote_mark", "line_first"]
LABEL = {"after_quote_tok": "token right AFTER a quote mark", "subword_cont": "subword continuation",
         "stopword": "stopword", "line_last": "last token of line", "capitalized": "capitalized word",
         "numeric": "contains digit", "schema_word": "meta-vocabulary (pattern/domain/…)",
         "punct_only": "punctuation only", "in_quote": "inside quoted span",
         "newline": "newline", "quote_mark": "quote mark \"", "line_first": "first token of line"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (key, name) in zip(axes, (("v3rl", "v3 RL"), ("v3ws", "v3 warm-start"))):
    r = res[key]
    ys = np.arange(len(ORDER))[::-1]
    for y, cat in zip(ys, ORDER):
        if cat not in r:
            continue
        d = r[cat]
        sig = d["lo"] > 0 or d["hi"] < 0
        col = "#2ca02c" if (sig and d["excess"] > 0) else "#d62728" if sig else "#999999"
        ax.errorbar(d["excess"], y, xerr=[[d["excess"] - d["lo"]], [d["hi"] - d["excess"]]],
                    fmt="o", ms=6, color=col, capsize=3, lw=1.8)
        ax.text(0.0155, y, f"{d['frac']:.0%}", fontsize=8, color="#555", va="center")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(ys); ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=10)
    ax.set_xlabel("excess marginal ΔFVE vs other tokens\n(index-stratified; 95% bootstrap CI)")
    ax.set_title(f"{name}  (grey % = token share)")
    ax.grid(alpha=.3, axis="x")
    ax.set_xlim(-0.05, 0.045)
fig.suptitle("Which token categories carry more/less marginal FVE? (index-controlled)", y=1.0, fontsize=13)
fig.tight_layout()
out = f"{D}/fig_token_categories.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
