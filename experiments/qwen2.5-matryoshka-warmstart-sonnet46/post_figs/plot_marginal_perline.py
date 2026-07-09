"""fig6: average marginal FVE per list line (each model's own lines).
fig7: same for ours, but kitft as a 10-equal-word-chunk control (needs
chunks_fve_t1_kitft.csv). Neutral labels."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OURS, BASE = "matryoshka NLA (ours)", "kitft baseline"
C_OURS, C_BASE = "#9467bd", "#888888"
K = 10


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    rows = [(int(r[list(r)[0]]), float(r["fve"])) for r in csv.DictReader(open(p))]
    return dict(rows)


def marginal(d, kmax=K):
    ks = [k for k in range(1, kmax + 1) if k in d]
    prev = 0.0
    out = []
    for k in ks:
        out.append(d[k] - prev); prev = d[k]
    return ks, out


ours = load("lines_fve_t1_iter0000200.csv")
kit = load("lines_fve_t1_kitft.csv")
if ours and kit:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ks, m = marginal(ours)
    ax.plot(ks, m, "o-", color=C_OURS, lw=2.4, ms=6, label=f"{OURS} — per line")
    ks2, m2 = marginal(kit)
    ax.plot(ks2, m2, "s--", color=C_BASE, lw=2.0, ms=6, label=f"{BASE} — per line")
    ax.axhline(0, color="k", lw=.8, alpha=.5)
    ax.set_xlabel("list line index")
    ax.set_ylabel("additional FVE from this line (ΔFVE)")
    ax.set_title("Average marginal FVE per list line")
    ax.set_xticks(range(1, K + 1)); ax.grid(alpha=.3); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig6_marginal_perline.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig6 ok")

chunks = load("chunks_fve_t1_kitft.csv")
if ours and chunks:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ks, m = marginal(ours)
    ax.plot(ks, m, "o-", color=C_OURS, lw=2.4, ms=6, label=f"{OURS} — per line")
    ks3, m3 = marginal(chunks)
    ax.plot(ks3, m3, "s--", color=C_BASE, lw=2.0, ms=6,
            label=f"{BASE} — per tenth of its output (word chunks)")
    ax.axhline(0, color="k", lw=.8, alpha=.5)
    ax.set_xlabel("line index (ours)  /  word-chunk index (baseline)")
    ax.set_ylabel("additional FVE from this segment (ΔFVE)")
    ax.set_title("Marginal FVE per segment — length-matched control")
    ax.set_xticks(range(1, K + 1)); ax.grid(alpha=.3); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig7_marginal_perline_chunked.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig7 ok")
else:
    print("fig7 skipped (chunks CSV missing)")

# ---- solo (no-control) variant ----
if ours:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ks, m = marginal(ours, kmax=9)
    ax.bar(ks, m, color=C_OURS, alpha=.85, width=0.72)
    for k, v in zip(ks, m):
        ax.text(k, v + (0.006 if v >= 0 else -0.014), f"{v:.3f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="k", lw=.8, alpha=.5)
    ax.set_xlabel("list line index")
    ax.set_ylabel("additional FVE from this line (ΔFVE)")
    ax.set_title("Average marginal FVE per list line")
    ax.set_xticks(range(1, 10)); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{HERE}/fig6_marginal_perline_solo.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig6 solo ok")


# ---- solo log variant (lines 1-9) ----
if ours:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ks, m = marginal(ours, kmax=9)
    ax.bar(ks, m, color=C_OURS, alpha=.85, width=0.72, log=True)
    for k, v in zip(ks, m):
        ax.text(k, v * 1.12, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xlabel("list line index")
    ax.set_ylabel("additional FVE from this line (ΔFVE, log)")
    ax.set_title("Average marginal FVE per list line")
    ax.set_xticks(range(1, 10)); ax.grid(alpha=.3, axis="y", which="both")
    fig.tight_layout(); fig.savefig(f"{HERE}/fig6_marginal_perline_solo_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("fig6 solo log ok")
