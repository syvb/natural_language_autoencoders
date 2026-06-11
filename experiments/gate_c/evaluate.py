"""Gate C evaluation: residualized contrasts + doc-clustered bootstrap.

Pre-registered (STATE.md):
- Nuisance model: ridge v18 -> d, fit on train-split rows only (lambda picked
  on an inner doc-level split of train), applied to holdout rows.
  residual r = d - ridge(v18).
- Primary contrast per diff arm X in {armA_8k, armB_8k}:
  paired per-position Delta_i = cos(pred_X_i, r_i) - cos(pred_0p_i, r_i)
  on the shared eval positions.
- Stats: doc-clustered bootstrap (cluster = doc_idx), 10k resamples,
  percentile CIs; Holm over the 2 contrasts.
- Decision: PASS = Holm-significant (CI_lo > 0) AND point >= +0.05.
  MARGINAL = +0.02..+0.05 or CI straddles 0. FAIL = within +/-0.02.
- Robustness: partial alignment beyond arm0p — project pred_0p direction out
  of d per position, score arms against that target.
- armA_full vs armA_8k: descriptive data-size slope only.

Outputs eval_results.json + printed table.
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
ARMS = ["arm0p_8k", "armA_8k", "armB_8k", "armA_full"]
N_BOOT = 10_000
SEED = 0


def cluster_boot(values, docs, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the mean of `values`, resampling doc clusters."""
    docs = np.asarray(docs)
    uniq = np.unique(docs)
    sums = np.zeros(len(uniq))
    cnts = np.zeros(len(uniq))
    idx = np.searchsorted(uniq, docs)
    np.add.at(sums, idx, values)
    np.add.at(cnts, idx, 1)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[picks].sum(1) / cnts[picks].sum(1)
    return means


def cosine(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1)
                              * np.linalg.norm(b, axis=-1) + 1e-9)


def main():
    meta = json.loads((WORK / "meta2.json").read_text())
    v18 = np.load(WORK / "acts18.npy", mmap_mode="r")
    d = np.load(WORK / "diff_targets.npy", mmap_mode="r")
    split = np.array([m["split"] for m in meta])
    doc = np.array([m["doc_idx"] for m in meta])
    tr = np.where(split == "train")[0]
    ho = np.where(split == "holdout")[0]

    # --- ridge nuisance model: v18 -> d, lambda from inner doc split of train
    X = np.asarray(v18[tr], dtype=np.float64)
    Y = np.asarray(d[tr], dtype=np.float64)
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
        print(f"[ridge] lambda {lam:g}  inner-val R2 {r2:.4f}", flush=True)
        if best is None or r2 > best[1]:
            best = (lam, r2)
    lam = best[0]
    mx, my = X.mean(0), Y.mean(0)
    W = np.linalg.solve((X - mx).T @ (X - mx) + lam * np.eye(X.shape[1]),
                        (X - mx).T @ (Y - my))
    Xh = np.asarray(v18[ho], dtype=np.float64)
    Dh = np.asarray(d[ho], dtype=np.float64)
    ridge_pred = (Xh - mx) @ W + my
    resid = Dh - ridge_pred
    r2_h = 1 - ((Dh - ridge_pred) ** 2).sum() / ((Dh - Dh.mean(0)) ** 2).sum()
    print(f"[ridge] chosen lambda {lam:g}  holdout R2 {r2_h:.4f}")
    np.save(WORK / "resid_holdout.npy", resid.astype(np.float32))
    np.save(WORK / "resid_holdout_rows.npy", ho)
    ho_row = {int(t): k for k, t in enumerate(ho)}

    # --- per-arm predictions, mapped to holdout rows
    arm = {}
    for name in ARMS:
        p = WORK / f"preds_{name}.npy"
        if not p.exists():
            print(f"[eval] missing {p}, skipping {name}")
            continue
        preds = np.load(p).astype(np.float64)
        tidx = np.load(WORK / f"tidx_{name}.npy")
        keep = np.array([int(t) in ho_row for t in tidx])
        rows = np.array([ho_row[int(t)] for t in tidx[keep]])
        arm[name] = {"tidx": tidx[keep], "rows": rows, "preds": preds[keep],
                     "cos_d": cosine(preds[keep], Dh[rows]),
                     "cos_r": cosine(preds[keep], resid[rows])}
        print(f"[eval] {name}: n={keep.sum()}  cos_d {arm[name]['cos_d'].mean():.4f}"
              f"  cos_resid {arm[name]['cos_r'].mean():.4f}")

    out = {"ridge_lambda": lam, "ridge_holdout_R2": r2_h, "arms": {}, "contrasts": {}}
    for name, a in arm.items():
        docs_a = doc[ho[a["rows"]]]
        bd = cluster_boot(a["cos_d"], docs_a)
        br = cluster_boot(a["cos_r"], docs_a)
        out["arms"][name] = {
            "n": int(len(a["rows"])),
            "cos_d": a["cos_d"].mean(), "cos_d_ci": np.percentile(bd, [2.5, 97.5]).tolist(),
            "cos_resid": a["cos_r"].mean(), "cos_resid_ci": np.percentile(br, [2.5, 97.5]).tolist(),
        }

    # --- primary paired contrasts vs arm0p_8k on shared positions
    base = arm.get("arm0p_8k")
    pvals = {}
    for name in ["armA_8k", "armB_8k"]:
        if name not in arm or base is None:
            continue
        a = arm[name]
        shared = np.intersect1d(a["tidx"], base["tidx"])
        ia = {int(t): k for k, t in enumerate(a["tidx"])}
        ib = {int(t): k for k, t in enumerate(base["tidx"])}
        ka = np.array([ia[int(t)] for t in shared])
        kb = np.array([ib[int(t)] for t in shared])
        delta = a["cos_r"][ka] - base["cos_r"][kb]
        docs_s = np.array([doc[int(t)] for t in shared])
        boots = cluster_boot(delta, docs_s)
        p_one = float((boots <= 0).mean())
        pvals[name] = p_one
        out["contrasts"][name] = {
            "n_shared": int(len(shared)),
            "delta": float(delta.mean()),
            "delta_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_one_sided": p_one,
        }

    # Holm over the two contrasts
    order = sorted(pvals, key=pvals.get)
    for rank, name in enumerate(order):
        adj = min(1.0, pvals[name] * (len(order) - rank))
        if rank > 0:
            adj = max(adj, out["contrasts"][order[rank - 1]]["p_holm"])
        out["contrasts"][name]["p_holm"] = adj

    for name, c in out["contrasts"].items():
        sig = c["p_holm"] < 0.05
        pt = c["delta"]
        lo, hi = c["delta_ci"]
        if sig and pt >= 0.05:
            verdict = "PASS"
        elif pt >= 0.02:
            verdict = "MARGINAL"
        elif pt <= -0.02 and hi < 0:
            verdict = "WORSE"
        else:
            verdict = "FAIL(null)"
        c["verdict"] = verdict

    # --- robustness: partial alignment beyond arm0p (project pred_0p out of d)
    if base is not None:
        ib = {int(t): k for k, t in enumerate(base["tidx"])}
        for name, a in arm.items():
            shared = [int(t) for t in a["tidx"] if int(t) in ib]
            ka = np.array([k for k, t in enumerate(a["tidx"]) if int(t) in ib])
            kb = np.array([ib[t] for t in shared])
            u = base["preds"][kb]
            u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
            dd = Dh[a["rows"][ka]]
            t_perp = dd - (dd * u).sum(-1, keepdims=True) * u
            out["arms"][name]["cos_beyond0p"] = float(
                cosine(a["preds"][ka], t_perp).mean())

    (WORK / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
