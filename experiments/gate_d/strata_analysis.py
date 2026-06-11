#!/usr/bin/env python3
"""Position-stratified analysis of Gate D held-out critic predictions.

Pure local compute. Reads existing arrays, writes strata_analysis.json
(strata_analysis.md is written separately from these numbers).

Definitions
-----------
cos_<arm>[k]  : cosine(pred_arm[k], resid_attn_holdout[k]) in float64.
d_TV          : cos_armT - cos_armV   (attention-trace evidence vs raw context tail)
d_OV          : cos_armO - cos_armV   (oracle evidence vs raw context tail)
d_VC          : cos_armV - cos_armC   (verbatim context vs paraphrase context)

Covariates (per holdout position i = meta row index)
----------------------------------------------------
write_norm    : ||attn_out[i]||_2
dist_mass     : contribution-weighted attention mass at sources j with
                1 <= j <= pos-16 (i.e. >15 tokens before pos; j=0 sink excluded).
                weights c[j] = sum_h (head_norms[i,h]/sum_h' head_norms[i,h']) * attn_rows[i,h,j]
max_dist_w    : max_j c[j] over the same distant-j set
top_head_frac : max_h head_norms[i,h] / sum_h head_norms[i,h]
resid_frac    : ||resid[i]|| / ||attn_out[i]||
pos           : position in doc
ctx_len       : len(meta.context_tail) in characters
self_mass     : c[pos]  (exploratory)
local_mass    : sum c[j] for pos-15 <= j <= pos-1  (exploratory)
attn_entropy  : entropy of c[1:pos+1] renormalized  (exploratory)

Inference: doc-clustered bootstrap (resample the 553 holdout doc_idx clusters
with replacement, 5000 resamples) for every CI reported.
"""
import json
import math
import numpy as np

RNG = np.random.default_rng(20260611)
N_BOOT = 5000
DIST_GAP = 15  # "distant" = more than 15 tokens before pos

# ---------------------------------------------------------------- load
rows = np.load("resid_attn_rows.npy")                      # meta row per resid row
resid = np.load("resid_attn_holdout.npy").astype(np.float64)
resid_norm = np.linalg.norm(resid, axis=1)
meta = json.load(open("meta_d.json"))
n = len(rows)
pos_of = {int(r): k for k, r in enumerate(rows)}

ARMS = ["armC_8k", "armT_8k", "armH_8k", "armZ_8k", "armV_8k", "armO_8k"]
cos = {}   # arm -> float64 array of len n, NaN where arm missing the position
for arm in ARMS:
    p = np.load(f"preds_{arm}.npy").astype(np.float64)
    t = np.load(f"tidx_{arm}.npy")
    idx = np.array([pos_of[int(x)] for x in t])
    c = np.full(n, np.nan)
    pn = np.linalg.norm(p, axis=1)
    c[idx] = (p * resid[idx]).sum(1) / (pn * resid_norm[idx])
    cos[arm] = c

cT, cV, cO, cC, cH, cZ = (cos[a] for a in
                          ["armT_8k", "armV_8k", "armO_8k", "armC_8k", "armH_8k", "armZ_8k"])
d_TV = cT - cV
d_OV = cO - cV
d_VC = cV - cC  # NaN on 17 positions armC lacks

# ---------------------------------------------------------------- covariates
attn_out = np.load("attn_out.npy", mmap_mode="r")
head_norms = np.load("head_norms.npy", mmap_mode="r")
attn_rows = np.load("attn_rows.npy", mmap_mode="r")

write_norm = np.empty(n)
dist_mass = np.empty(n)
max_dist_w = np.empty(n)
top_head_frac = np.empty(n)
self_mass = np.empty(n)
local_mass = np.empty(n)
attn_entropy = np.empty(n)
pos_arr = np.empty(n)
ctx_len = np.empty(n)
doc_idx = np.empty(n, dtype=np.int64)
tok_at = []

for k, mi in enumerate(rows):
    mi = int(mi)
    mrow = meta[mi]
    p = mrow["pos"]
    pos_arr[k] = p
    ctx_len[k] = len(mrow["context_tail"])
    doc_idx[k] = mrow["doc_idx"]
    tok_at.append(mrow["token_at_pos"])
    write_norm[k] = np.linalg.norm(np.asarray(attn_out[mi], dtype=np.float64))
    hn = np.asarray(head_norms[mi], dtype=np.float64)
    hw = hn / hn.sum()
    top_head_frac[k] = hw.max()
    A = np.asarray(attn_rows[mi], dtype=np.float64)        # [28, 512]
    c_src = hw @ A                                         # contribution-weighted row [512]
    dist_hi = p - DIST_GAP                                 # exclusive upper bound: j <= p-16
    dseg = c_src[1:dist_hi] if dist_hi > 1 else np.empty(0)
    dist_mass[k] = dseg.sum()
    max_dist_w[k] = dseg.max() if dseg.size else 0.0
    self_mass[k] = c_src[p]
    local_mass[k] = c_src[max(1, p - DIST_GAP):p].sum()
    seg = c_src[1:p + 1]
    q = seg / seg.sum()
    q = q[q > 0]
    attn_entropy[k] = float(-(q * np.log(q)).sum())

resid_frac = resid_norm / write_norm

COVS = {
    "write_norm": write_norm,
    "dist_mass": dist_mass,
    "max_dist_w": max_dist_w,
    "top_head_frac": top_head_frac,
    "resid_frac": resid_frac,
    "pos": pos_arr,
    "ctx_len": ctx_len,
}
COVS_EXPLORE = {
    "self_mass": self_mass,
    "local_mass": local_mass,
    "attn_entropy": attn_entropy,
}

# ---------------------------------------------------------------- bootstrap machinery
docs = np.unique(doc_idx)
n_docs = len(docs)
doc_of = np.searchsorted(docs, doc_idx)                    # 0..n_docs-1 per row
BOOT_DRAWS = RNG.integers(0, n_docs, size=(N_BOOT, n_docs))

def cluster_boot_mean(values, mask):
    """Doc-clustered bootstrap CI of mean(values[mask]). values may contain NaN."""
    v = np.where(mask & ~np.isnan(values), values, 0.0)
    w = (mask & ~np.isnan(values)).astype(np.float64)
    sums = np.bincount(doc_of, weights=v, minlength=n_docs)
    cnts = np.bincount(doc_of, weights=w, minlength=n_docs)
    bs = sums[BOOT_DRAWS].sum(1)
    bc = cnts[BOOT_DRAWS].sum(1)
    ok = bc > 0
    means = bs[ok] / bc[ok]
    est = sums.sum() / cnts.sum()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(est), float(lo), float(hi), int(w.sum())

def decile_table(cov, d, boot_extremes=True):
    """Mean of d per decile of cov. CI (clustered) on deciles 1 and 10."""
    valid = ~np.isnan(d)
    edges = np.quantile(cov[valid], np.linspace(0, 1, 11))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    dec = np.clip(np.digitize(cov, edges) - 1, 0, 9)
    out = []
    for q in range(10):
        mask = (dec == q) & valid
        entry = {"decile": q + 1, "n": int(mask.sum()),
                 "cov_lo": float(np.quantile(cov[mask], 0.0)) if mask.any() else None,
                 "cov_hi": float(np.quantile(cov[mask], 1.0)) if mask.any() else None,
                 "mean": float(d[mask].mean()) if mask.any() else None}
        if boot_extremes and q in (0, 9):
            est, lo, hi, _ = cluster_boot_mean(d, mask)
            entry["ci95"] = [lo, hi]
        out.append(entry)
    return out

def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

def pearson(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

# ---------------------------------------------------------------- strata counting
strata_examined = 0
results = {"n_holdout": n, "n_docs": int(n_docs), "n_boot": N_BOOT,
           "headline": {}}

for arm in ARMS:
    est, lo, hi, cnt = cluster_boot_mean(cos[arm], ~np.isnan(cos[arm]))
    results["headline"][f"cos_{arm}"] = {"mean": est, "ci95": [lo, hi], "n": cnt}
for name, d in [("d_TV", d_TV), ("d_OV", d_OV), ("d_VC", d_VC)]:
    est, lo, hi, cnt = cluster_boot_mean(d, ~np.isnan(d))
    results["headline"][name] = {"mean": est, "ci95": [lo, hi], "n": cnt}

# ---------------------------------------------------------------- 2/3: decile tables
results["deciles_d_TV"] = {}
results["deciles_d_OV"] = {}
for cname, cov in COVS.items():
    results["deciles_d_TV"][cname] = decile_table(cov, d_TV)
    results["deciles_d_OV"][cname] = decile_table(cov, d_OV)
    strata_examined += 20

# fraction of positions where T beats V, and where O beats V
results["frac_pos"] = {
    "d_TV_gt0": float((d_TV > 0).mean()),
    "d_OV_gt0": float((d_OV > 0).mean()),
}

# focused confirmatory slice (pre-registered in the task): top-decile dist_mass
edges = np.quantile(dist_mass, np.linspace(0, 1, 11))
top_dm = dist_mass >= edges[9]
for name, d in [("d_TV", d_TV), ("d_OV", d_OV)]:
    est, lo, hi, cnt = cluster_boot_mean(d, top_dm)
    results[f"top_decile_dist_mass_{name}"] = {"mean": est, "ci95": [lo, hi], "n": cnt}

# ---------------------------------------------------------------- 4: armV dominance correlates
results["correlates"] = {}
ALL_COVS = {**COVS, **COVS_EXPLORE}
for target_name, target in [("d_VC", d_VC), ("cos_V", cV), ("d_TV", d_TV), ("d_OV", d_OV)]:
    results["correlates"][target_name] = {
        cname: {"pearson": pearson(cov, target), "spearman": spearman(cov, target)}
        for cname, cov in ALL_COVS.items()
    }

# ---------------------------------------------------------------- 5: token-class strata
FUNC_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "not", "no", "nor", "so", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "us", "our",
    "you", "your", "i", "me", "my", "who", "whom", "whose", "which", "what",
    "when", "where", "why", "how", "there", "here", "all", "any", "some",
    "each", "both", "more", "most", "other", "into", "over", "under", "up",
    "down", "out", "off", "about", "again", "also", "just", "only", "very",
    "s", "t", "re", "ve", "ll", "d", "m",
}

def tok_class(t):
    if "\n" in t:
        return "newline"
    s = t.strip()
    if s == "":
        return "space"
    if all(not ch.isalnum() for ch in s):
        return "punct"
    if s.isdigit():
        return "number"
    if s.lower() in FUNC_WORDS:
        return "function"
    if s.isalpha():
        return "content"
    return "other"

tclass = np.array([tok_class(t) for t in tok_at])
results["token_class"] = {}
for cls in sorted(set(tclass)):
    mask = tclass == cls
    entry = {"n": int(mask.sum())}
    for name, d in [("d_TV", d_TV), ("d_OV", d_OV), ("d_VC", d_VC),
                    ("cos_V", cV), ("cos_T", cT)]:
        est, lo, hi, cnt = cluster_boot_mean(d, mask)
        entry[name] = {"mean": est, "ci95": [lo, hi]}
    results["token_class"][cls] = entry
    strata_examined += 2  # d_TV and d_OV strata formally examined

# ---------------------------------------------------------------- 6: exploration
explore = {}

# 6a. exploratory covariate deciles for d_TV
explore["deciles_d_TV_exploratory"] = {}
for cname, cov in COVS_EXPLORE.items():
    explore["deciles_d_TV_exploratory"][cname] = decile_table(cov, d_TV)
    strata_examined += 10

# 6b. interaction: top-decile dist_mass restricted to content tokens
mask = top_dm & (tclass == "content")
est, lo, hi, cnt = cluster_boot_mean(d_TV, mask)
explore["top_dm_and_content_d_TV"] = {"mean": est, "ci95": [lo, hi], "n": cnt}
est, lo, hi, cnt = cluster_boot_mean(d_OV, mask)
explore["top_dm_and_content_d_OV"] = {"mean": est, "ci95": [lo, hi], "n": cnt}
strata_examined += 2

# 6c. d_TV stratified by cos_armC (a third arm: proxy for "context predictability",
# avoids selecting directly on V's or T's own noise). quintiles.
ok = ~np.isnan(cC)
qedges = np.quantile(cC[ok], np.linspace(0, 1, 6))
qedges[0] -= 1e-9; qedges[-1] += 1e-9
qq = np.clip(np.digitize(cC, qedges) - 1, 0, 4)
tab = []
for q in range(5):
    mask = (qq == q) & ok
    est, lo, hi, cnt = cluster_boot_mean(d_TV, mask)
    tab.append({"quintile": q + 1, "cos_C_range": [float(qedges[q]), float(qedges[q + 1])],
                "n": cnt, "mean_d_TV": est, "ci95": [lo, hi],
                "mean_cos_T": float(np.nanmean(cT[mask])),
                "mean_cos_V": float(np.nanmean(cV[mask]))})
    strata_examined += 1
explore["d_TV_by_cosC_quintile"] = tab

# 6d. how much of cos_V is "shared with everything" — cos between arm preds?
# mean pairwise cosine of predictions (sampled) to see if V vs T differ in target
# coverage or just amplitude. Use 500 random holdout rows.
sub = RNG.choice(n, size=500, replace=False)
pV = np.load("preds_armV_8k.npy").astype(np.float64)[sub]
pT = np.load("preds_armT_8k.npy").astype(np.float64)[sub]
pO = np.load("preds_armO_8k.npy").astype(np.float64)[sub]
def mcos(a, b):
    return float(((a * b).sum(1) / (np.linalg.norm(a, 1e-30 + 0*a[:,0], axis=1) if False else np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).mean())
explore["pred_similarity"] = {
    "cos(predT,predV)": mcos(pT, pV),
    "cos(predO,predV)": mcos(pO, pV),
    "cos(predT,predO)": mcos(pT, pO),
    "n_sampled": 500,
}

# 6e. norm-prediction skill: corr(|pred| , |resid|) per arm — do arms differ in
# predicting the *size* vs the *direction* of the write?
explore["norm_corr"] = {}
for arm in ["armT_8k", "armV_8k", "armO_8k", "armC_8k"]:
    p = np.load(f"preds_{arm}.npy").astype(np.float64)
    t = np.load(f"tidx_{arm}.npy")
    idx = np.array([pos_of[int(x)] for x in t])
    explore["norm_corr"][arm] = {
        "pearson_prednorm_residnorm": pearson(np.linalg.norm(p, axis=1), resid_norm[idx]),
    }

# 6f. best and worst strata scan summary (for multiplicity framing): collect all
# decile means computed for d_TV and report the max.
allTV = []
for cname in list(COVS) :
    for e in results["deciles_d_TV"][cname]:
        allTV.append((cname, e["decile"], e["mean"]))
for cname in COVS_EXPLORE:
    for e in explore["deciles_d_TV_exploratory"][cname]:
        allTV.append((cname, e["decile"], e["mean"]))
best = max(allTV, key=lambda x: x[2])
explore["best_d_TV_decile_anywhere"] = {"covariate": best[0], "decile": best[1], "mean": best[2]}

results["explore"] = explore
results["strata_examined_total"] = strata_examined
results["covariate_summary"] = {
    cname: {"mean": float(np.mean(cov)), "p10": float(np.quantile(cov, .1)),
            "median": float(np.median(cov)), "p90": float(np.quantile(cov, .9))}
    for cname, cov in ALL_COVS.items()
}

with open("strata_analysis.json", "w") as f:
    json.dump(results, f, indent=1)
print("strata examined:", strata_examined)
print(json.dumps(results["headline"], indent=1))
