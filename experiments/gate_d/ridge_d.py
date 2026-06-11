"""Gate D0.2: content-confound ridge v_pre -> attn_out (pre-registered gate).

GO if holdout R^2 < 0.40; CAUTION 0.40-0.60 (use residualized target
downstream); STOP > 0.60. Also saves resid_attn_holdout.npy for later
phases. Same machinery as gate_c evaluate.py.
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
SEED = 0

meta = json.loads((WORK / "meta_d.json").read_text())
X_all = np.load(WORK / "v_pre.npy", mmap_mode="r")
Y_all = np.load(WORK / "attn_out.npy", mmap_mode="r")
split = np.array([m["split"] for m in meta])
doc = np.array([m["doc_idx"] for m in meta])
tr = np.where(split == "train")[0]
ho = np.where(split == "holdout")[0]

X = np.asarray(X_all[tr], dtype=np.float64)
Y = np.asarray(Y_all[tr], dtype=np.float64)
rng = np.random.default_rng(SEED)
tr_docs = np.unique(doc[tr])
val_docs = set(rng.choice(tr_docs, len(tr_docs) // 5, replace=False).tolist())
inner_val = np.array([dd in val_docs for dd in doc[tr]])
Xf, Yf = X[~inner_val], Y[~inner_val]
Xv, Yv = X[inner_val], Y[inner_val]
mx, my = Xf.mean(0), Yf.mean(0)
G = (Xf - mx).T @ (Xf - mx)
B = (Xf - mx).T @ (Yf - my)
best = None
for lam in [1e2, 1e3, 1e4, 1e5, 1e6]:
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B)
    P = (Xv - mx) @ W + my
    r2 = 1 - ((Yv - P) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum()
    print(f"[ridge_d] lambda {lam:g}  inner-val R2 {r2:.4f}", flush=True)
    if best is None or r2 > best[1]:
        best = (lam, r2)
lam = best[0]
mx, my = X.mean(0), Y.mean(0)
W = np.linalg.solve((X - mx).T @ (X - mx) + lam * np.eye(X.shape[1]),
                    (X - mx).T @ (Y - my))
Xh = np.asarray(X_all[ho], dtype=np.float64)
Yh = np.asarray(Y_all[ho], dtype=np.float64)
pred = (Xh - mx) @ W + my
r2_h = 1 - ((Yh - pred) ** 2).sum() / ((Yh - Yh.mean(0)) ** 2).sum()
np.save(WORK / "resid_attn_holdout.npy", (Yh - pred).astype(np.float32))
np.save(WORK / "resid_attn_rows.npy", ho)
gate = "GO" if r2_h < 0.40 else ("CAUTION" if r2_h <= 0.60 else "STOP")
print(f"[ridge_d] chosen lambda {lam:g}  holdout R2 {r2_h:.4f}  -> {gate}")
Path(WORK / "ridge_d_result.json").write_text(json.dumps(
    {"lambda": lam, "holdout_R2": r2_h, "gate": gate}))
