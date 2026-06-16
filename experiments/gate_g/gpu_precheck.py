"""Gate G headroom pre-check on GPU (instant). Same logic as precheck.py:
linear h_20 -> h_22 residual = upper bound on what the text channel can add."""
import json
from pathlib import Path
import numpy as np
import torch

GC = Path("/work/gate_c")
meta = json.loads((GC / "meta2.json").read_text())
split = np.array([m["split"] for m in meta])
tr = np.where(split == "train")[0]
ho = np.where(split == "holdout")[0]

dev = "cuda"
H20 = torch.tensor(np.load(GC / "acts20.npy"), device=dev, dtype=torch.float32)
H22 = torch.tensor(np.load(GC / "acts22.npy"), device=dev, dtype=torch.float32)


def cos_rows(X, Y):
    return torch.nn.functional.cosine_similarity(X, Y, dim=1)


def headroom(X, Y):
    raw = cos_rows(X, Y).mean().item()
    delta = Y - X
    rel = (delta.norm(dim=1) / Y.norm(dim=1)).mean().item()
    cosxd = cos_rows(X, delta).mean().item()
    xm, ym = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = X[tr] - xm, Y[tr] - ym
    G = Xc.T @ Xc
    XtY = Xc.T @ Yc
    Xho = X[ho] - xm
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4):
        W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev), XtY)
        Yhat = Xho @ W + ym
        c = cos_rows(Y[ho], Yhat).mean().item()
        if best is None or c > best[1]:
            best = (lam, c)
    return raw, rel, cosxd, best


raw, rel, cosxd, (lam, c) = headroom(H20, H22)
print(f"positions: {len(tr)} train / {len(ho)} holdout")
print(f"raw cos(h20,h22)        = {raw:+.4f}")
print(f"rel |delta|/|h22|       = {rel:.4f}")
print(f"cos(h20, delta)         = {cosxd:+.4f}   (high => diff is mostly rescale of h20)")
print(f"linear recover cos      = {c:+.4f}  (lam={lam:g})")
print(f">> LINEAR HEADROOM      = {1-c:.4f}   <- upper bound on text value-add")
print("rule: >0.08 comfortable | 0.03-0.08 marginal | <0.03 widen the gap")
