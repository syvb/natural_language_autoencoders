#!/usr/bin/env python3
"""Turn a held-out eval rollout dump into a readable per-sample table.

Run ON the box (reads torch dumps). For each penalty TAG, writes
/workspace/heldout_results/<TAG>/samples.jsonl with one row per held-out sample:
  doc_id, n_tokens(activation pos), response_len, reconstruction MSE, cos_sim, fve_est, explanation, source_text
and prints aggregate FVE.

reward in the dump is -MSE on L2-normalised vectors (penalty off), so
  mse = -reward ; cos = 1 - mse/2 ; fve_est = 1 - mse / VAR  (VAR≈0.72 predict-mean baseline)
Usage: python extract_heldout.py <TAG> [<TAG> ...]   (or no args = all under heldout_results)
"""
import glob
import json
import statistics
import sys
from pathlib import Path

import torch

VAR = 0.72  # canonical predict-the-mean baseline (training notes); fve_est is approximate


def _samples(dump):
    d = torch.load(dump, weights_only=False)
    return d["samples"] if isinstance(d, dict) and "samples" in d else d


def _get(s, k, default=None):
    return s.get(k, default) if isinstance(s, dict) else getattr(s, k, default)


def process(tag):
    base = Path("/workspace/heldout_results") / tag
    dumps = sorted(glob.glob(str(base / "rollout" / "*.pt")))
    if not dumps:
        print(f"{tag}: no rollout dump"); return
    rows, mses = [], []
    for dp in dumps:
        for s in _samples(dp):
            reward = _get(s, "reward")
            if reward is None:
                continue
            md = _get(s, "metadata", {}) or {}
            resp = _get(s, "response", "") or ""
            expl = resp.split("<explanation>")[-1].split("</explanation>")[0].strip() if "<explanation>" in resp else resp.strip()
            mse = -float(reward)
            mses.append(mse)
            rows.append({
                "doc_id": md.get("doc_id"),
                "response_len": _get(s, "response_length"),
                "mse": round(mse, 5),
                "cos_sim": round(1 - mse / 2, 5),
                "fve_est": round(1 - mse / VAR, 5),
                "explanation": expl,
                "source_text": md.get("detokenized_text_truncated"),
            })
    out = base / "samples.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    agg_fve = 1 - statistics.mean(mses) / VAR if mses else float("nan")
    mean_len = statistics.mean([r["response_len"] for r in rows if r["response_len"]]) if rows else 0
    print(f"{tag}: {len(rows)} samples -> {out} | mean_len={mean_len:.0f}tok agg_FVE={agg_fve:.3f} "
          f"mean_cos={statistics.mean([r['cos_sim'] for r in rows]):.3f}")


if __name__ == "__main__":
    tags = sys.argv[1:] or [Path(p).name for p in glob.glob("/workspace/heldout_results/*") if Path(p).is_dir()]
    for t in tags:
        process(t)
