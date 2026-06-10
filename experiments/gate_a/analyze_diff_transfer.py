"""Diff-direction transfer probe (Gate A addendum).

Does the unmodified L20 NLA pipeline already carry information about the
layer-to-layer CHANGE? For a layer pair (X, Y), compare the reconstruction
difference against the true diff:

    cos( AR(text_Y) - AR(text_X),  v_Y - v_X )

The AR is never given a diff as input — both reconstructions are ordinary
text->vector outputs, subtracted numerically. Because the subtraction lives
in the AR's layer-20-biased output space, the score is a LOWER BOUND on the
diff information present in the explanation texts. Null control: the
reconstruction diff of two independent explanations of the SAME vector
(L20 vs L20_rep), which isolates text-sampling noise.

Also reports retrieval (recon picks its own gold among all positions) for
full vectors and for diffs.
"""

import json

import numpy as np

acts = np.load("acts.npy")
recons = np.load("recons.npy").astype(np.float32)
labels = json.loads(open("recons_labels.json").read())
R = {l: recons[i] for i, l in enumerate(labels)}
n = acts.shape[0]

def cosv(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)

def retrieval(pred, gold):
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    g = gold / np.linalg.norm(gold, axis=1, keepdims=True)
    S = p @ g.T
    ranks = (S > S.diagonal()[:, None]).sum(1) + 1
    return {"top1": float((ranks == 1).mean()), "top5": float((ranks <= 5).mean()),
            "median_rank": int(np.median(ranks)), "n": int(n)}

out = {"pairs": {}, "full_vector_retrieval": {}}

print("=== diff-direction transfer: cos(recon_Y - recon_X, v_Y - v_X) ===")
for lx, ly in [(20, 21), (20, 22), (20, 23), (20, 24), (19, 21), (18, 22), (17, 23)]:
    gd = acts[:, ly] - acts[:, lx]
    c = cosv(R[f"L{ly}"] - R[f"L{lx}"], gd)
    null = cosv(R["L20_rep"] - R["L20"], gd)
    entry = {"mean_cos": float(c.mean()), "std": float(c.std()),
             "frac_positive": float((c > 0).mean()),
             "null_mean_cos": float(null.mean()),
             "diff_retrieval": retrieval(R[f"L{ly}"] - R[f"L{lx}"], gd)}
    out["pairs"][f"L{lx}->L{ly}"] = entry
    print(f"  L{lx}->L{ly} (gap {ly - lx}): mean {c.mean():+.3f}  "
          f"frac>0 {(c > 0).mean():.2f}  null {null.mean():+.3f}  "
          f"diff top-1 retrieval {entry['diff_retrieval']['top1']:.2f}")

print("\n=== full-vector retrieval (recon identifies its position among all golds) ===")
for L in [16, 20, 21, 24]:
    out["full_vector_retrieval"][f"L{L}"] = retrieval(R[f"L{L}"], acts[:, L])
    r = out["full_vector_retrieval"][f"L{L}"]
    print(f"  L{L}: top-1 {r['top1']:.2f}  median rank {r['median_rank']}/{n}")

res = json.load(open("results.json"))
res["_diff_transfer"] = out
json.dump(res, open("results.json", "w"), indent=2)
print("\nresults.json updated (_diff_transfer)")
