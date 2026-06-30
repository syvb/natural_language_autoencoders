"""One diagram per held-out example: each line of the AV's explanation + the
round-trip FVE achieved when the explanation is truncated to that line.

Reads fve_by_line.json (from fve_by_line.py). For each example renders a two-panel
figure: left = the numbered explanation lines, right = a horizontal bar of the
cumulative FVE if you stop after that line. Also writes a combined overview
(FVE vs line index, all examples). Run locally; matplotlib only.
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(RES, "fve_by_line.json")
d = json.load(open(DATA))
exs = d["examples"]


def wrap(s, w=84):
    s = s.replace("\n", " ")
    return s if len(s) <= w else s[: w - 1] + "…"


def color(f):
    return "#2ca02c" if f >= 0 else "#d62728"


for n, ex in enumerate(exs):
    lines = ex["lines"]; fves = ex["fve_by_k"]; N = len(lines)
    y = np.arange(N)
    h = max(3.2, 0.34 * N + 1.2)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, h), gridspec_kw={"width_ratios": [3.1, 1.5]})
    # left: line text
    axL.set_xlim(0, 1); axL.set_ylim(N - 0.5, -0.5); axL.axis("off")
    for i, ln in enumerate(lines):
        axL.text(0.0, i, f"{i+1:>2}. {wrap(ln)}", va="center", ha="left", family="monospace", fontsize=8.3)
    axL.set_title(f"AV explanation lines (doc `{ex['doc_id']}`)", fontsize=10, loc="left")
    # right: cumulative FVE bars
    axR.barh(y, fves, color=[color(f) for f in fves], height=0.72)
    axR.set_ylim(N - 0.5, -0.5)
    lo = min(0.0, min(fves)) - 0.05
    axR.set_xlim(lo, 1.0)
    axR.axvline(0, color="k", lw=0.8)
    axR.axvline(fves[-1], color="#555", ls="--", lw=1.0)
    axR.set_yticks(y); axR.set_yticklabels([str(i + 1) for i in range(N)], fontsize=7)
    axR.set_xlabel("round-trip FVE (explanation truncated to line k)")
    axR.grid(axis="x", alpha=0.3)
    for i, f in enumerate(fves):
        axR.text(f + (0.012 if f >= 0 else -0.012), i, f"{f:.2f}", va="center",
                 ha="left" if f >= 0 else "right", fontsize=6.6, color="#222")
    axR.set_title(f"full-list FVE = {fves[-1]:.2f}  (1 line = {fves[0]:.2f})", fontsize=9.5)
    tail = (ex.get("source_tail") or "")[-180:]
    fig.suptitle(f"Information-upfront: FVE vs truncation line  —  source tail: …{wrap(tail,120)}",
                 y=1.005, fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(RES, f"fve_by_line_ex{n}_{ex['doc_id'].replace('/', '_').replace(':', '_')}.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}  (N={N}, FVE 1->{fves[0]:.2f}  full->{fves[-1]:.2f})")

# combined overview
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for n, ex in enumerate(exs):
    f = ex["fve_by_k"]
    ax.plot(range(1, len(f) + 1), f, "o-", ms=3, lw=1.6, alpha=0.85,
            label=f"{ex['doc_id'][:34]} (N={len(f)})")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("explanation truncated to first k lines")
ax.set_ylabel("round-trip FVE")
ax.set_title("FVE rises and saturates within the first few lines (v2 NLA round-trip)")
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fve_by_line_overview.png"), dpi=130, bbox_inches="tight")
print("wrote fve_by_line_overview.png")
