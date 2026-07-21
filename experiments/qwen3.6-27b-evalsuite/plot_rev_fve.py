"""FVE vs token truncation with the standard NLA's unit order REVERSED.

Tests whether the standard NLA's late-rising truncation curve reflects
backloaded information: if so, reversing its unit order should make it rise
early like the matryoshka's.

Matryoshka reference = checked-in results/clean_token_fve.csv (same protocol,
same seed-42 held-out docs). Std curves = fresh same-seed rerun
(results/rev_token_fve_std_{orig,revlines,revsents}.csv); the orig curve
doubles as a replication of clean_std_token_fve.csv.
Writes results/fve_truncation_reversed.png.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results"
PURPLE, GRAY, RED, ORANGE = "#9467bd", "#888888", "#d62728", "#e8923a"


def read_csv(path):
    with open(path) as fh:
        rd = list(csv.reader(fh))
    return [(int(a), float(b)) for a, b, _ in rd[1:]]


def main():
    mat = read_csv(f"{RES}/clean_token_fve.csv")
    orig = read_csv(f"{RES}/rev_token_fve_std_orig.csv")
    revl = read_csv(f"{RES}/rev_token_fve_std_revlines.csv")
    revs_p = f"{RES}/rev_token_fve_std_revsents.csv"
    revs = read_csv(revs_p) if os.path.exists(revs_p) else None
    prev = read_csv(f"{RES}/clean_std_token_fve.csv")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(*zip(*mat), "-", color=PURPLE, lw=2, label="matryoshka (trained order)")
    ax.plot(*zip(*prev), "-", color="#cccccc", lw=1.2,
            label="standard, original order (prior run)")
    ax.plot(*zip(*orig), "--", color=GRAY, lw=2, label="standard, original order")
    ax.plot(*zip(*revl), "-", color=RED, lw=2, label="standard, lines reversed")
    if revs:
        ax.plot(*zip(*revs), ":", color=ORANGE, lw=2,
                label="standard, sentences reversed")

    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("explanation truncation (content tokens)")
    ax.set_ylabel("round-trip FVE (own critic)")
    ax.set_xlim(0, 260)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Does the standard NLA backload information?\n"
                 "FVE vs token truncation with its unit order reversed "
                 "(Qwen3.6-27B, clean held-out)", fontsize=11.5)

    sp = f"{RES}/rev_summary.json"
    note = ""
    if os.path.exists(sp):
        s = json.load(open(sp))
        note = "  |  ".join(f"{k.replace('|full', '')} full-len {v:.3f}"
                            for k, v in sorted(s.items()))
    fig.text(0.01, 0.005,
             "100 fresh held-out L42 activations (Ultra-FineWeb idx 300000+), "
             f"seed-42 selection identical to the original curves.\n{note}",
             fontsize=7.0, color="#888888")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = f"{RES}/fve_truncation_reversed.png"
    fig.savefig(out, dpi=170)
    print(f"[saved] {out}")

    # pared-down two-curve version: matryoshka vs standard-lines-reversed
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(*zip(*mat), "-", color=PURPLE, lw=2.2, label="matryoshka (trained order)")
    ax.plot(*zip(*revl), "-", color=RED, lw=2.2, label="standard, lines reversed")
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("explanation truncation (content tokens)")
    ax.set_ylabel("round-trip FVE (own critic)")
    ax.set_xlim(0, 260)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Matryoshka vs standard NLA read back-to-front\n"
                 "FVE vs token truncation (Qwen3.6-27B, clean held-out)",
                 fontsize=11.5)
    fig.text(0.01, 0.005,
             "100 fresh held-out L42 activations (Ultra-FineWeb idx 300000+), "
             "seed-42 selection identical to the original curves.",
             fontsize=7.0, color="#888888")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out2 = f"{RES}/fve_truncation_mat_vs_stdrev.png"
    fig.savefig(out2, dpi=170)
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
