"""Build KL-judge candidate diff predictions from the scored samples.

Selection uses cos(AR(text), residual) — the residualized metric. The KL
judge is the independent evaluator, so selection bias on the AR metric is
controlled by construction.
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
acts = np.load(WORK / "acts.npy")
residuals = np.load(WORK / "residuals.npy")

bo = np.load(WORK / "bo_recons.npy").astype(np.float32)
ok = np.load(WORK / "bo_ok.npy")
conds = json.loads((WORK / "bo_conds.json").read_text())
B = {c: bo[i] for i, c in enumerate(conds)}        # [N, 8, d]
OKB = {c: ok[i] for i, c in enumerate(conds)}

r2 = np.load(WORK / "round2_recons.npy").astype(np.float32)
conds2 = json.loads((WORK / "round2_conds.json").read_text())
R2 = {c: r2[i] for i, c in enumerate(conds2)}

def pick_best(cond):
    rec, m = B[cond], OKB[cond]
    rn = rec / np.clip(np.linalg.norm(rec, axis=2, keepdims=True), 1e-9, None)
    tn = residuals / np.linalg.norm(residuals, axis=1, keepdims=True)
    scores = (rn * tn[:, None, :]).sum(2)
    scores[~m] = -np.inf
    idx = scores.argmax(1)
    return rec[np.arange(rec.shape[0]), idx], idx

out = {}
for cond in ["dav8", "resav8"]:
    best, idx = pick_best(cond)
    np.save(WORK / f"cand_{cond}_bo8.npy", best.astype(np.float32))
    np.save(WORK / f"cand_{cond}_first.npy", B[cond][:, 0].astype(np.float32))
    out[f"{cond}_bo8"] = f"cand_{cond}_bo8.npy"
    out[f"{cond}_first"] = f"cand_{cond}_first.npy"
    json.dump(idx.tolist(), open(WORK / f"sel_idx_{cond}.json", "w"))

np.save(WORK / "cand_dav_t07.npy", R2["diffav_18_22"].astype(np.float32))
np.save(WORK / "cand_claude_v2.npy", R2["real_18_22_v2"].astype(np.float32))
out["dav_t07"] = "cand_dav_t07.npy"
out["claude_v2"] = "cand_claude_v2.npy"
out["content_baseline"] = "ga_l20_recons.npy"
json.dump(out, open(WORK / "kl_candidates.json", "w"))
print("candidates:", list(out))
