"""Gate G pre-check ($0, local CPU).

The Stage-0 hypothesis: AR(h_L, AV-text-of-h_{L+k}) reconstructs h_{L+k}
better than AR(h_L) alone, because the text carries h_{L+k} information that
h_L lacks. The *ceiling* on that value-add is the headroom: the part of
h_{L+k} not recoverable from h_L. A best-case LINEAR map h_L -> h_{L+k} gives
an UPPER bound on headroom (a nonlinear AR can only do better at predicting
h_{L+k} from h_L, leaving the text even less to do). If linear headroom is
already tiny, Stage 0 has nothing to measure and we should widen the gap.

We have h_18, h_20, h_22 at 15k aligned positions (gate_c), doc-level split.
Reports raw geometry + linear headroom for gap-2 (20->22, the chosen target),
gap-4 (18->22), and gap-2-shallow (18->20).
"""
import json
from pathlib import Path
import numpy as np

GC = Path(__file__).resolve().parent.parent / "gate_c"
meta = json.loads((GC / "meta2.json").read_text())
split = np.array([m["split"] for m in meta])
# subsample train for the ridge fit — slow reference BLAS here; 5k rows is
# plenty for a direction-recovery estimate and keeps each gap under ~2 min.
_rng = np.random.default_rng(0)
_tr_all = np.where(split == "train")[0]
tr = _rng.choice(_tr_all, size=min(5000, len(_tr_all)), replace=False)
ho = np.where(split == "holdout")[0]

A = {L: np.load(GC / f"acts{L}.npy") for L in (18, 20, 22)}


def cos_rows(X, Y):
    xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return (xn * yn).sum(1)


def report(name, src, dst):
    X, Y = A[src], A[dst]
    delta = Y - X
    raw = cos_rows(X, Y).mean()
    rel = (np.linalg.norm(delta, axis=1) / np.linalg.norm(Y, axis=1)).mean()
    cos_x_d = cos_rows(X, delta).mean()            # is the diff just "more of h_L"?
    # gram matrices ONCE; ridge-solve per lambda is cheap
    xm, ym = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = X[tr] - xm, Y[tr] - ym
    G = Xc.T @ Xc
    XtY = Xc.T @ Yc
    Xho = X[ho] - xm
    best = None
    for lam in (1e2, 1e3):
        Gl = G.copy()
        Gl[np.diag_indices_from(Gl)] += lam
        W = np.linalg.solve(Gl, XtY)
        Yhat = Xho @ W + ym
        c = cos_rows(Y[ho], Yhat).mean()
        rf = (np.linalg.norm(Y[ho] - Yhat, axis=1)
              / (np.linalg.norm(Y[ho], axis=1) + 1e-8)).mean()
        if best is None or c > best[1]:
            best = (lam, c, rf)
    lam, c, rf = best
    print(f"\n=== {name}: h_{src} -> h_{dst} ===")
    print(f"  raw cos(h_{src}, h_{dst})          = {raw:+.4f}")
    print(f"  rel diff norm |delta|/|h_{dst}|     = {rel:.4f}")
    print(f"  cos(h_{src}, delta)  (rescale-ness) = {cos_x_d:+.4f}")
    print(f"  linear recover cos (lam={lam:g})     = {c:+.4f}")
    print(f"  >> LINEAR HEADROOM (1 - cos)       = {1 - c:.4f}   (upper bound on text value-add)")
    print(f"  residual norm frac                 = {rf:.4f}")
    return 1 - c


if __name__ == "__main__":
    print(f"positions: {tr.sum()} train / {ho.sum()} holdout")
    h_target = report("CHOSEN TARGET (gap 2)", 20, 22)
    h_gap4 = report("gap 4", 18, 22)
    report("gap 2 shallow", 18, 20)
    print("\n" + "=" * 56)
    print(f"Chosen target (20->22) headroom ceiling: {h_target:.3f}")
    print(f"Wider gap   (18->22) headroom ceiling: {h_gap4:.3f}")
    print("Rule of thumb: >0.08 = comfortable room for Stage 0 to detect")
    print("               0.03-0.08 = marginal, consider wider gap")
    print("               <0.03 = too little, widen the gap")
