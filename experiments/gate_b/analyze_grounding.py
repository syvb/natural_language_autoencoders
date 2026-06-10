"""Gate B step 3 (local): are Claude's diff labels grounded in the true diff?

For each real condition, cos(AR(diff_text), v_Y - v_X) with controls:
  - shuffled: same texts scored against other positions' gold diffs
  - null texts: hallucinated "diffs" (same-vector pairs) scored against the
    real golds
Also: retrieval, verdict-conditioned means, and comparison against Gate A's
recon-subtraction baseline (AR(text_Y) - AR(text_X)).
"""

import json

import numpy as np

acts = np.load("/home/debian/gate_a/acts.npy")
ga_recons = np.load("/home/debian/gate_a/recons.npy").astype(np.float32)
ga_labels = json.loads(open("/home/debian/gate_a/recons_labels.json").read())
GA = {l: ga_recons[i] for i, l in enumerate(ga_labels)}

dr = np.load("diff_recons.npy").astype(np.float32)
ok = np.load("diff_recons_ok.npy")
conds = json.loads(open("diff_recons_conds.json").read())
DR = {c: dr[i] for i, c in enumerate(conds)}
OK = {c: ok[i] for i, c in enumerate(conds)}
labels = json.loads(open("gate_b_labels.json").read())
n = acts.shape[0]
rng = np.random.default_rng(0)
perm = rng.permutation(n)

def cosv(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)

def retrieval(pred, gold, mask):
    p = pred[mask] / np.linalg.norm(pred[mask], axis=1, keepdims=True)
    g = gold[mask] / np.linalg.norm(gold[mask], axis=1, keepdims=True)
    S = p @ g.T
    ranks = (S > S.diagonal()[:, None]).sum(1) + 1
    return (ranks == 1).mean(), np.median(ranks), mask.sum()

out = {}
for cond, (lx, ly) in [("real_19_21", (19, 21)), ("real_18_22", (18, 22))]:
    gold = acts[:, ly] - acts[:, lx]
    m = OK[cond]
    c = cosv(DR[cond], gold)
    c_shuf = cosv(DR[cond], gold[perm])
    c_null = cosv(DR["null_20_20"], gold)
    mn = m & OK["null_20_20"]
    base = cosv(GA[f"L{ly}"] - GA[f"L{lx}"], gold)
    verd = np.array([labels[cond][i]["verdict"] == "different" for i in range(n)])
    top1, med, k = retrieval(DR[cond], gold, m)
    out[cond] = {
        "cos_matched": float(c[m].mean()),
        "cos_shuffled": float(c_shuf[m].mean()),
        "cos_null_texts": float(c_null[mn].mean()),
        "cos_recon_subtraction_baseline": float(base.mean()),
        "retrieval_top1": float(top1), "retrieval_median_rank": int(med),
        "n_ok": int(k),
        "cos_when_verdict_different": float(c[m & verd].mean()),
        "cos_when_verdict_same": float(c[m & ~verd].mean()),
        "frac_positive": float((c[m] > 0).mean()),
    }
    e = out[cond]
    print(f"=== {cond} (n={e['n_ok']}) ===")
    print(f"  cos(AR(diff_text), true diff):    {e['cos_matched']:+.3f}  (frac>0 {e['frac_positive']:.2f})")
    print(f"  shuffled-position control:        {e['cos_shuffled']:+.3f}")
    print(f"  null-pair texts vs same golds:    {e['cos_null_texts']:+.3f}")
    print(f"  recon-subtraction baseline:       {e['cos_recon_subtraction_baseline']:+.3f}")
    print(f"  diff retrieval: top-1 {e['retrieval_top1']:.2f}, median rank {e['retrieval_median_rank']}/{n}")
    print(f"  by verdict: different {e['cos_when_verdict_different']:+.3f} | same {e['cos_when_verdict_same']:+.3f}")
    print()

json.dump(out, open("grounding_results.json", "w"), indent=2)
print("grounding_results.json written")
