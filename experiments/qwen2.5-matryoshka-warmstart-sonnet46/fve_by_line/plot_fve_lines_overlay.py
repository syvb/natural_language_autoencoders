"""Single-panel FVE-by-line diagram: first 10 lines, FVE as bar width, the AV
explanation text overlaid on the bars. No header. One PNG per example.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
d = json.load(open(os.path.join(RES, "fve_by_line.json")))

K = 10
BAR = "#cfe8f5"        # light blue bar
BAR_EDGE = "#5b9bd5"
TXT = "#10242e"        # dark charcoal-teal text, reads on bar + white
VAL = "#1f6f8b"        # FVE value color


def trunc(s, n=96):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


for n, ex in enumerate(d["examples"]):
    lines = ex["lines"][:K]
    fves = ex["fve_by_k"][:K]
    N = len(lines)
    y = list(range(N))
    valx = max(fves) + 0.16          # fixed right-hand value column
    xmax = valx + 0.03
    fig, ax = plt.subplots(figsize=(13.5, 0.40 * N + 0.45))

    ax.barh(y, fves, height=0.74, color=BAR, edgecolor=BAR_EDGE, linewidth=0.8, zorder=2)
    for i, (ln, f) in enumerate(zip(lines, fves)):
        ax.plot([f, valx - 0.05], [i, i], ls=":", lw=0.8, color="#9bbed6", zorder=1)  # leader
        ax.text(0.012, i, f"{i+1}.  {trunc(ln)}", va="center", ha="left",
                fontsize=10.3, color=TXT, zorder=4)
        ax.text(valx, i, f"{f:.2f}", va="center", ha="right",
                fontsize=10.5, fontweight="bold", color=VAL, zorder=4)

    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.set_xticks([t / 10 for t in range(0, int(max(fves) * 10) + 2)])
    ax.set_xlabel("round-trip FVE  (explanation truncated to this line)", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    out = os.path.join(RES, f"fve_lines_overlay_ex{n}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}  (doc {ex['doc_id']}, FVE@1={fves[0]:.2f} @{K}={fves[-1]:.2f})")
