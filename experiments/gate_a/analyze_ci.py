"""Cluster-bootstrap (by document) 95% CIs for per-layer FVE from saved
AR reconstructions. Also a paired CI for the L21-vs-L20 FVE difference.

FVE per (re)sample set S: 1 - sum_i 2(1-cos_i) / sum_i ||u_i - mean_S(u)||^2
on unit vectors u (direction-only, mse_scale cancels). Both numerator and
denominator are recomputed per bootstrap resample.
"""

import json

import numpy as np

acts = np.load("acts.npy")  # [180, 29, 3584] fp32
recons = np.load("recons.npy").astype(np.float32)  # [n_labels, 180, 3584]
labels = json.loads(open("recons_labels.json").read())
meta = json.loads(open("meta.json").read())
res = json.load(open("results.json"))

doc_ids = np.array([m["doc_idx"] for m in meta])
docs = np.unique(doc_ids)
doc_rows = [np.where(doc_ids == d)[0] for d in docs]
N_BOOT = 10_000
rng = np.random.default_rng(0)

units = acts / np.linalg.norm(acts, axis=2, keepdims=True)  # [180, 29, d]
recon_units = recons / np.linalg.norm(recons, axis=2, keepdims=True)

# per-sample cosine per label
cos = {}
for li, label in enumerate(labels):
    L = int(label.removeprefix("L").removesuffix("_rep"))
    cos[label] = (recon_units[li] * units[:, L]).sum(axis=1)  # [180]

def fve_of(rows: np.ndarray, c: np.ndarray, u: np.ndarray) -> float:
    cs, us = c[rows], u[rows]
    denom = (np.linalg.norm(us - us.mean(axis=0), axis=1) ** 2).sum()
    return 1.0 - 2.0 * (1.0 - cs).sum() / denom

# bootstrap doc resamples (shared across labels -> enables paired diffs)
resamples = [np.concatenate([doc_rows[j] for j in rng.integers(0, len(docs), len(docs))])
             for _ in range(N_BOOT)]

out = {}
boot_fves = {}
for label in labels:
    L = int(label.removeprefix("L").removesuffix("_rep"))
    c, u = cos[label], units[:, L]
    point = fve_of(np.arange(len(meta)), c, u)
    boots = np.array([fve_of(rows, c, u) for rows in resamples])
    boot_fves[label] = boots
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out[label] = {"fve": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                  "cos_mean": float(cos[label].mean()),
                  "cos_ci": [float(x) for x in np.percentile(
                      [cos[label][rows].mean() for rows in resamples], [2.5, 97.5])]}
    print(f"{label:>9}: FVE {point:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

d = boot_fves["L21"] - boot_fves["L20"]
d_lo, d_hi = np.percentile(d, [2.5, 97.5])
point_d = out["L21"]["fve"] - out["L20"]["fve"]
out["_L21_minus_L20"] = {"diff": float(point_d), "ci_lo": float(d_lo),
                         "ci_hi": float(d_hi), "p_gt_0": float((d > 0).mean())}
print(f"\npaired L21-L20 FVE diff: {point_d:+.3f}  [{d_lo:+.3f}, {d_hi:+.3f}]  "
      f"P(diff>0) = {(d > 0).mean():.4f}")

for label, e in out.items():
    if not label.startswith("_"):
        res[label].update({"fve_nrm": e["fve"], "fve_ci95": [e["ci_lo"], e["ci_hi"]],
                           "cos_ci95": e["cos_ci"]})
res["_L21_minus_L20"] = out["_L21_minus_L20"]
res["_ci_method"] = (f"cluster bootstrap over {len(docs)} docs, {N_BOOT} resamples, "
                     "percentile 95% CI; FVE numerator+denominator recomputed per resample")
json.dump(res, open("results.json", "w"), indent=2)
print("\nresults.json updated")
