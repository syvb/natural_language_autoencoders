"""Figure for the prompt-induced secrets audit.

Left panel : main audit arms (transcripts +/- explanations), mean trait grade
             with exact-ID counts; secret words are 0 in every arm (noted).
Right panel: truncation sweep (explanations-only + token identity), mean
             trait grade vs number of explanation units shown.

Usage: python plot_secrets.py <dir with sa_audit_scores.json + sa_trunc_scores.json>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results/secrets"
S = json.load(open(f"{D}/sa_audit_scores.json"))
T = json.load(open(f"{D}/sa_trunc_scores.json"))

C = {"mat": "#2a78d6", "std": "#eb6834"}
GRAY = "#898781"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
NAME = {"mat": "matryoshka", "std": "standard"}
TRAITS = [o for o, v in S.items() if v["kind"] == "trait"]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), facecolor=SURF)

# ── left: main arms ──────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(SURF)
ARMS = [("bb", "transcripts\nonly", GRAY),
        ("plc", "+ WRONG organism's\nexplanations (placebo)", "#bdbbb4"),
        ("std", "+ standard\nexplanations", C["std"]),
        ("mat", "+ matryoshka\nexplanations", C["mat"])]
for i, (arm, label, color) in enumerate(ARMS):
    gs = [S[o]["arms"][arm]["grade"] for o in TRAITS]
    v = sum(gs) / (2 * len(gs))
    nex = sum(g == 2 for g in gs)
    ax.bar(i, v, width=0.62, color=color, zorder=3)
    ax.annotate(f"{v:.0%}\n{nex}/6 exact", (i, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([a[1] for a in ARMS], fontsize=8.5, color=INK)
ax.set_title("what the auditor recovers, by evidence given\n"
             "(6 trait organisms; secret words: 0% in every arm)",
             fontsize=10, loc="left", color=INK)

# ── right: truncation sweep ──────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(SURF)
KLS = ["k1", "k2", "k4", "full"]
X = range(len(KLS))
for m in ("mat", "std"):
    ys = []
    for kl in KLS:
        rs = [r for r in T if r["kind"] == "trait" and r["model"] == m and r["k"] == kl]
        ys.append(sum(r["grade"] for r in rs) / (2 * len(rs)))
    ax.plot(X, ys, color=C[m], lw=2, marker="o", ms=7, zorder=3, label=NAME[m])
    ax.annotate(f"{ys[-1]:.0%}", (len(KLS) - 1, ys[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=C[m], fontweight="bold")
ax.set_xticks(list(X))
ax.set_xticklabels(["1 unit", "2 units", "4 units", "full"], fontsize=9, color=INK)
ax.set_xlim(-0.3, len(KLS) - 0.4)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("explanation truncation (no transcripts; token at each\n"
             "position always shown) — mat leads at every length",
             fontsize=10, loc="left", color=INK)

for ax in axes:
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
axes[0].set_ylabel("mean audit grade (fraction of max)", color=INK, fontsize=9)

fig.suptitle("Auditing prompt-induced secrets with NLA explanations "
             "(Qwen3.6-27B, L42; auditor: nex-n2-mini)",
             x=0.02, ha="left", fontsize=12, color=INK)
fig.text(0.02, 0.012,
         "12 organisms (4 secret words, 6 covert traits, 2 controls) × 4 probe conversations · "
         "grades: 2 specific / 1 partial / 0 miss, median of 3 judge calls · both controls false-positive in every arm",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.045, 1, 0.90))
fig.savefig(f"{D}/fig_secrets_audit.png", dpi=170, facecolor=SURF)
print(f"saved {D}/fig_secrets_audit.png")
