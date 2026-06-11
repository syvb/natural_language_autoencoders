"""Gate D2 evaluation (gate_c evaluate.py adapted per code review M4).

- Targets: attn_out.npy; residual target LOADED from ridge_d.py outputs
  (resid_attn_holdout.npy / resid_attn_rows.npy) — not recomputed.
- Baseline arm: armC (content reads of v_pre). Contrasts: armT, armH, armZ
  vs armC; Holm over however many are present.
- Pre-registered: PASS = Holm CI>0 AND delta >= +0.05; MARGINAL +0.02..0.05;
  FAIL within +/-0.02; doc-clustered bootstrap 10k.
- Robustness: alignment beyond armC (project armC's prediction direction
  out of the raw target per position).
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
BASE = "armC_8k"
ARMS = ["armC_8k", "armT_8k", "armH_8k", "armZ_8k", "armT2_8k"]
N_BOOT = 10_000
SEED = 0


def cluster_boot(values, docs, seed=SEED):
    docs = np.asarray(docs)
    uniq = np.unique(docs)
    sums = np.zeros(len(uniq)); cnts = np.zeros(len(uniq))
    k = np.searchsorted(uniq, docs)
    np.add.at(sums, k, values); np.add.at(cnts, k, 1)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    return sums[picks].sum(1) / cnts[picks].sum(1)


def cosine(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1)
                              * np.linalg.norm(b, axis=-1) + 1e-9)


def main():
    meta = json.loads((WORK / "meta_d.json").read_text())
    Y = np.load(WORK / "attn_out.npy", mmap_mode="r")
    resid = np.load(WORK / "resid_attn_holdout.npy").astype(np.float64)
    ho = np.load(WORK / "resid_attn_rows.npy")
    doc = np.array([m["doc_idx"] for m in meta])
    ho_row = {int(t): k for k, t in enumerate(ho)}

    arm = {}
    for name in ARMS:
        p = WORK / f"preds_{name}.npy"
        if not p.exists():
            print(f"[eval_d] missing {p}, skipping")
            continue
        preds = np.load(p).astype(np.float64)
        tidx = np.load(WORK / f"tidx_{name}.npy")
        assert len(set(tidx.tolist())) == len(tidx), f"{name}: dup tidx"
        keep = np.array([int(t) in ho_row for t in tidx])
        rows = np.array([ho_row[int(t)] for t in tidx[keep]])
        Dh = np.asarray(Y[ho[rows]], dtype=np.float64)
        arm[name] = {"tidx": tidx[keep], "rows": rows, "preds": preds[keep],
                     "cos_d": cosine(preds[keep], Dh),
                     "cos_r": cosine(preds[keep], resid[rows])}
        print(f"[eval_d] {name}: n={keep.sum()} cos_d {arm[name]['cos_d'].mean():.4f}"
              f" cos_resid {arm[name]['cos_r'].mean():.4f}")

    out = {"arms": {}, "contrasts": {}}
    for name, a in arm.items():
        docs_a = doc[ho[a["rows"]]]
        bd = cluster_boot(a["cos_d"], docs_a)
        br = cluster_boot(a["cos_r"], docs_a)
        out["arms"][name] = {
            "n": int(len(a["rows"])),
            "cos_d": a["cos_d"].mean(),
            "cos_d_ci": np.percentile(bd, [2.5, 97.5]).tolist(),
            "cos_resid": a["cos_r"].mean(),
            "cos_resid_ci": np.percentile(br, [2.5, 97.5]).tolist(),
        }

    base = arm.get(BASE)
    pvals = {}
    for name, a in arm.items():
        if name == BASE or base is None:
            continue
        shared = np.intersect1d(a["tidx"], base["tidx"])
        ia = {int(t): k for k, t in enumerate(a["tidx"])}
        ib = {int(t): k for k, t in enumerate(base["tidx"])}
        delta = (a["cos_r"][[ia[int(t)] for t in shared]]
                 - base["cos_r"][[ib[int(t)] for t in shared]])
        boots = cluster_boot(delta, np.array([doc[int(t)] for t in shared]))
        pvals[name] = float((boots <= 0).mean())
        out["contrasts"][name] = {
            "n_shared": int(len(shared)),
            "delta": float(delta.mean()),
            "delta_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "p_one_sided": pvals[name],
        }

    order = sorted(pvals, key=pvals.get)
    for rank, name in enumerate(order):
        adj = min(1.0, pvals[name] * (len(order) - rank))
        if rank > 0:
            adj = max(adj, out["contrasts"][order[rank - 1]]["p_holm"])
        out["contrasts"][name]["p_holm"] = adj
        c = out["contrasts"][name]
        lo, hi = c["delta_ci"]
        pt = c["delta"]
        if c["p_holm"] < 0.05 and pt >= 0.05:
            v = "PASS"
        elif pt >= 0.02:
            v = "MARGINAL"
        elif pt <= -0.02 and hi < 0:
            v = "WORSE"
        else:
            v = "FAIL(null)"
        c["verdict"] = v

    if base is not None:
        ib = {int(t): k for k, t in enumerate(base["tidx"])}
        for name, a in arm.items():
            ka = np.array([k for k, t in enumerate(a["tidx"]) if int(t) in ib])
            kb = np.array([ib[int(t)] for t in a["tidx"] if int(t) in ib])
            u = base["preds"][kb]
            u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
            Dh = np.asarray(Y[ho[a["rows"][ka]]], dtype=np.float64)
            t_perp = Dh - (Dh * u).sum(-1, keepdims=True) * u
            out["arms"][name]["cos_beyondC"] = float(
                cosine(a["preds"][ka], t_perp).mean())

    (WORK / "eval_d_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:3500])


if __name__ == "__main__":
    main()
