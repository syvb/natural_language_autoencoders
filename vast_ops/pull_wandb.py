#!/usr/bin/env python3
"""Pull wandb run histories for the length-penalty sweep into local CSVs.

This IS the persistence step — once run, experiment_results/ holds everything
needed to rebuild the length/FVE tradeoff chart without the GPU boxes.

Writes, under experiment_results/:
  <tag>_history.csv   full per-step history for each run (robust raw dump)
  summary.csv         one row/run: penalty, final + last-10 fve & response-len

Usage: WANDB_API_KEY=... python vast_ops/pull_wandb.py [--group G] [--project P]
(falls back to ~/.wandb_key)
"""
import argparse
import csv
import os
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiment_results"


def _find_col(cols, *patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="nla-length-penalty")
    ap.add_argument("--group", default=None)
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        kf = Path("~/.wandb_key").expanduser()
        if kf.exists():
            os.environ["WANDB_API_KEY"] = kf.read_text().strip()
    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    runs = api.runs(f"{entity}/{args.project}",
                    filters={"group": args.group} if args.group else None)
    OUT.mkdir(exist_ok=True)

    summary = []
    for run in runs:
        df = run.history(samples=100000, pandas=True)
        tag = run.name or run.id
        df.to_csv(OUT / f"{tag}_history.csv", index=False)
        cols = list(df.columns)
        fve_col = _find_col(cols, r"fve")
        len_col = _find_col(cols, r"response_length", r"resp.*len", r"\blen")
        m = re.search(r"pen([0-9.]+)", tag)
        penalty = float(m.group(1)) if m else float("nan")

        def last(col, n=1, agg="last"):
            if not col or col not in df:
                return ""
            s = df[col].dropna()
            if not len(s):
                return ""
            return round(float(s.tail(n).mean()), 5)

        summary.append({
            "tag": tag, "penalty": penalty,
            "fve_last": last(fve_col, 1), "fve_last10": last(fve_col, 10),
            "len_last": last(len_col, 1), "len_last10": last(len_col, 10),
            "fve_col": fve_col or "", "len_col": len_col or "",
        })
        print(f"pulled {tag:24s} fve_col={fve_col} len_col={len_col} rows={len(df)}")

    summary.sort(key=lambda r: (r["penalty"] != r["penalty"], r["penalty"]))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "penalty", "fve_last", "fve_last10",
                                          "len_last", "len_last10", "fve_col", "len_col"])
        w.writeheader()
        w.writerows(summary)
    print("wrote", OUT / "summary.csv")


if __name__ == "__main__":
    main()
