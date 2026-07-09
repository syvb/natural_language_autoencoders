"""Stage 2: frontloading generations (ratio stage cut from this run).

Frontload (port of caa_steering_v2/frontload_v2.py, trimmed R grid, T=1):
  a = b + r*||b||*v_hat for each trait x R_GRID x 40 bases -> explanation list.
Ratio (port of ratio_steering/gen_ratio.py, full design):
  a = b + ||b||*(r_y*v_yellow + r_s*v_syco), R in {0.6,0.9,1.3},
  lam = log2(r_y/r_s) over 9 points, 2 reps x 40 bases.

Injection here is EasyNLA's karvonen add-norm-match (direction-only), the
27B AVs' training convention — the analog of v3's injection_scale renorm.
Writes $WORK/suite/frontload_raw_$MODEL.json and ratio_raw_$MODEL.json.
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from suite_common import (CJK, MODEL, PARQUET, WORK, av_batch, items_of,
                          load_base, load_actor, prompt_ids)

R_GRID = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0]
TRAITS = ["sycophancy", "neuroticism", "yellow"]
RATIO_R = [0.6, 0.9, 1.3]
LAM_GRID = [-3.0, -2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0, 3.0]
REPS = 2
B = 16

V = np.load(f"{WORK}/suite/suite_dirs.npz")
bases = V["base_acts"]
units = {t: V[f"{t}_L42_genuine_unit"].astype(np.float32) for t in TRAITS}
vy, vs = units["yellow"], units["sycophancy"]

vref = [None]
actor, tok, cfg = load_actor(load_base(), vref)
pmsgs = pq.read_table(PARQUET, columns=["prompt"]).column("prompt")[0].as_py()
ids = prompt_ids(tok, cfg, pmsgs)
print(f"[{MODEL}] prompt len {len(ids)}", flush=True)

# ---- frontload conditions ----
fl_conds = []
for trait in TRAITS:
    u = units[trait]
    for r in R_GRID:
        for bi, b in enumerate(bases):
            bn = float(np.linalg.norm(b))
            fl_conds.append((trait, r, bi, (b + r * bn * u).astype(np.float32)))

fl_rows = []
for s in range(0, len(fl_conds), B):
    chunk = fl_conds[s:s + B]
    for (trait, r, bi, vec), g in zip(chunk, av_batch(actor, tok, vref, ids, [c[3] for c in chunk])):
        fl_rows.append({"trait": trait, "r": r, "base_idx": bi,
                        "items": items_of(g), "cjk": bool(CJK.search(g))})
    if (s // B) % 10 == 0:
        print(f"  frontload {min(s+B, len(fl_conds))}/{len(fl_conds)}", flush=True)
json.dump(fl_rows, open(f"{WORK}/suite/frontload_raw_{MODEL}.json", "w"), indent=1)
print(f"frontload done, cjk={sum(r['cjk'] for r in fl_rows)}/{len(fl_rows)}", flush=True)
print("GEN_DONE", flush=True)
