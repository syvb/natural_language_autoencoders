"""Figure: truncated-steering behavioural fidelity, matryoshka vs standard.

Two panels (one y-scale each, shared x): keep only the first k explanation
units, re-encode with the model's critic, patch at L42, regenerate seed-matched;
how well does the k-steer reproduce the ALL-units steer?
  left  = textual agreement (position-aligned token match; identical vectors
          give identical T=1 text, so this is a mechanical fidelity readout)
  right = semantic similarity (nex-n2-mini judge, 0-1)
Points annotated with the mean words actually read at that k.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = json.load(open("steer_fidelity.json"))["model"]
K = [1, 2, 3]
WORDS = {"mat": {1: 12, 2: 23, 3: 32}, "std": {1: 21, 2: 55, 3: 86}}
C = {"mat": "#2a78d6", "std": "#eb6834"}
NAME = {"mat": "matryoshka (lines)", "std": "standard (sentences)"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURF)
panels = [("textual", "textual agreement with full-explanation steer"),
          ("semantic", "semantic similarity (judge) with full steer")]
for ax, (key, title) in zip(axes, panels):
    ax.set_facecolor(SURF)
    ax.axhline(1.0, color=GRID, lw=1.2, ls="--", zorder=1)
    ax.text(0.06, 1.04, "1.0 = identical behaviour", color=INK2, fontsize=8,
            va="bottom", ha="left")
    for m in ("mat", "std"):
        ys = [F[m]["k"][str(k)][key] for k in K]
        ax.plot(K, ys, color=C[m], lw=2, marker="o", ms=8, zorder=3,
                markeredgecolor=SURF, markeredgewidth=1.5)
        up = (m == "mat")
        if key == "semantic":  # words only where the series don't cross
            for k, y in zip(K, ys):
                ax.annotate(f"{WORDS[m][k]} w", (k, y), textcoords="offset points",
                            xytext=(0, 12 if up else -19), ha="center",
                            fontsize=8, color=INK2)
        # direct label LEFT of the first point, where the series are far apart
        ax.annotate(NAME[m], (K[0], ys[0]), textcoords="offset points",
                    xytext=(-10, 6 if up else -6), ha="right",
                    va="bottom" if up else "top",
                    fontsize=9.5, color=C[m], fontweight="bold")
    ax.set_xticks(K)
    ax.set_xticklabels([f"first {k}" for k in K])
    ax.set_xlim(-0.35, 3.45)
    ax.set_ylim(0, 1.14)
    ax.set_xlabel("explanation units kept (k)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

fig.suptitle("Steering with a truncated explanation — patch v̂(first k units) back at L42, "
             "regenerate seed-matched (45 positions × 2 seeds)",
             color=INK, fontsize=12, x=0.02, ha="left")
fig.text(0.02, 0.015,
         "point labels = mean words read at that k · matryoshka full-FVE 0.65, standard 0.69 "
         "(same texts, own critics) · std first-sentence re-encoding FVE −0.57 (anti-informative)",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig("fig_steer_truncation_fidelity.png", dpi=170, facecolor=SURF)
print("saved fig_steer_truncation_fidelity.png")
