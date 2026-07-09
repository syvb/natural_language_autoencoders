"""Fold per-LINE eval-awareness into per-TOKEN weighted heatmap scores (matryoshka).

Reads eval_awareness_lines.json (per-line P(aware), saved by eval_awareness_lines.py
in precache iteration order) + precache.json, and adds two arrays per entry to
eval_awareness.json, aligned with `paware`:

  paware_fvew     — FVE-weighted mean P(aware):  Σ p_k·max(m_k,0) / Σ max(m_k,0)
                    (m_k = marginal FVE of line k → high-reconstruction lines dominate)
  paware_fvewnorm — same with rank-normalized weights max(m_k/mean_m[rank k], 0)
                    (removes the mechanical front-loading of marginal FVE)
  paware_lines    — per position, the raw list of per-line P(aware) (for the ΔFVE panel)

Alignment is verified by recomputing (marginal_fve, line_idx) per line and asserting
they match the saved arrays exactly.

    python eval_awareness_weighted.py space/precache.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("precache")
    args = ap.parse_args()
    here = Path(args.precache).parent
    entries = json.load(open(args.precache))["entries"]
    L = json.load(open(here / "eval_awareness_lines.json"))
    mfve = np.array(L["marginal_fve"]); paw = np.array(L["p_aware"]); lidx = np.array(L["line_idx"])

    # corpus-average marginal FVE per salience rank (for the normalized variant)
    mean_m = {k: mfve[lidx == k].mean() for k in np.unique(lidx)}
    print("rank-mean marginal FVE:", {int(k): round(float(v), 4) for k, v in mean_m.items()})

    ea_path = here / "eval_awareness.json"
    ea = json.load(open(ea_path))
    assert len(ea["entries"]) == len(entries)

    ptr = 0
    for e, ee in zip(entries, ea["entries"]):
        w1_arr, w2_arr, lines_arr = [], [], []
        for pos, r in enumerate(e["results"]):
            if not r or not r["lines"]:
                w1_arr.append(None); w2_arr.append(None); lines_arr.append(None); continue
            prev = 0.0
            p_k, m_k = [], []
            for k, (ln, f) in enumerate(zip(r["lines"], r["fve"])):
                # alignment check against the saved per-line arrays
                # saved marginal_fve was rounded to 5 dp → tolerance just above 5e-6
                assert abs(mfve[ptr] - (f - prev)) < 1e-4 and lidx[ptr] == k, \
                    f"misalignment at flat index {ptr} (entry pos {pos} line {k})"
                p_k.append(paw[ptr]); m_k.append(f - prev)
                prev = f; ptr += 1
            p_k = np.array(p_k); m_k = np.array(m_k)
            w1 = np.maximum(m_k, 0)
            w2 = np.maximum(m_k / np.array([mean_m[k] for k in range(len(m_k))]), 0)
            w1_arr.append(round(float((p_k * w1).sum() / w1.sum()), 5) if w1.sum() > 0 else None)
            w2_arr.append(round(float((p_k * w2).sum() / w2.sum()), 5) if w2.sum() > 0 else None)
            lines_arr.append([round(float(x), 4) for x in p_k])
        ee["paware_fvew"] = w1_arr
        ee["paware_fvewnorm"] = w2_arr
        ee["paware_lines"] = lines_arr
    assert ptr == len(paw), (ptr, len(paw))
    json.dump(ea, open(ea_path, "w"))
    flat = lambda key: [v for ee in ea["entries"] for v in ee[key] if v is not None]
    for key in ("paware", "paware_fvew", "paware_fvewnorm"):
        v = np.array(flat(key))
        print(f"{key:16s}: n={len(v)} mean={v.mean():.3f} p95={np.percentile(v,95):.3f} max={v.max():.3f}")
    print(f"[saved] {ea_path}")


if __name__ == "__main__":
    main()
