#!/usr/bin/env python3
"""Plot the length<->reconstruction tradeoff from per-run metrics.

Consumes experiment_results/summary.csv (one row per penalty run) and the
per-run experiment_results/<tag>/metrics.csv trajectories. Produces:
  - tradeoff.png : FVE (recon quality) vs mean response length, one point/penalty
  - trajectories.png : fve_nrm and mean_resp_len vs step, per penalty

summary.csv columns: tag,penalty,seed,fve_last,fve_last10,len_last,len_last10
metrics.csv columns: step,fve_nrm,mean_resp_len,median_resp_len,mean_reward,frac_failed,n_completed
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "experiment_results"


def _read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    summary_path = RESULTS / "summary.csv"
    if not summary_path.exists():
        sys.exit(f"no {summary_path} — run the parser first")
    rows = sorted(_read_csv(summary_path), key=lambda r: float(r["penalty"]))

    # ── tradeoff scatter ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [float(r["len_last10"]) for r in rows]
    ys = [float(r["fve_last10"]) for r in rows]
    pen = [float(r["penalty"]) for r in rows]
    ax.plot(xs, ys, "-o", color="#3060c0")
    for x, y, p in zip(xs, ys, pen):
        ax.annotate(f"λ={p:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    base = next((r for r in rows if float(r["penalty"]) == 0), None)
    if base:
        ax.axhline(float(base["fve_last10"]), ls="--", color="grey", lw=0.8, alpha=0.6)
        ax.annotate("control FVE", (max(xs), float(base["fve_last10"])), fontsize=8, color="grey")
    ax.set_xlabel("mean explanation length (response tokens, last-10-step avg)")
    ax.set_ylabel("reconstruction FVE (fve_nrm, last-10-step avg)")
    ax.set_title("Length penalty vs. reconstruction (continue-RL, Qwen2.5-7B NLA)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff.png", dpi=140)
    print("wrote", RESULTS / "tradeoff.png")

    # ── trajectories ───────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in rows:
        m = RESULTS / r["tag"] / "metrics.csv"
        if not m.exists():
            continue
        d = _read_csv(m)
        steps = [int(x["step"]) for x in d]
        a1.plot(steps, [float(x["fve_nrm"]) for x in d], label=f"λ={float(r['penalty']):g}")
        a2.plot(steps, [float(x["mean_resp_len"]) for x in d], label=f"λ={float(r['penalty']):g}")
    a1.set_xlabel("RL step"); a1.set_ylabel("fve_nrm"); a1.set_title("Reconstruction over training"); a1.grid(True, alpha=0.3); a1.legend(fontsize=8)
    a2.set_xlabel("RL step"); a2.set_ylabel("mean response length"); a2.set_title("Explanation length over training"); a2.grid(True, alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "trajectories.png", dpi=140)
    print("wrote", RESULTS / "trajectories.png")


if __name__ == "__main__":
    main()
