"""Gate C secondary analyses: KL judge significance + best-of-16 (descriptive).

KL: paired Wilcoxon signed-rank (normal approximation, fine at n~144) between
each arm's per-position KL and the content control's. The judge is
pre-registered as veto-only: an arm is vetoed if significantly WORSE.

Best-of-16: eval16 reads (16 diff-injection samples per holdout position)
scored by the trained armA critic against the ridge residual. Selection is
oracle (max cos vs the true residual), so we report a selection-bias control:
the same max-of-16 where each position's 16 candidates are scored against a
permuted (wrong-position) residual, averaged over 20 permutations.
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent


def wilcoxon(x, y):
    """Paired Wilcoxon signed-rank, normal approx, two-sided p."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    n = len(d)
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # midranks for ties
    a = np.abs(d)
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    w_pos = r[d > 0].sum()
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_pos - mu) / sd
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
    return z, p


def cosine(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1)
                              * np.linalg.norm(b, axis=-1) + 1e-9)


def main():
    out = {}

    # --- KL judge significance
    kl = json.loads((WORK / "kl_results.json").read_text())
    base = np.array(kl["pred_0p"])
    print(f"[kl] n={len(base)}")
    out["kl"] = {}
    for name in ["zero_diff", "pred_A", "pred_B"]:
        a = np.array(kl[name])
        z, p = wilcoxon(a, base)
        out["kl"][name] = {"mean": a.mean(), "median": float(np.median(a)),
                           "vs_pred_0p_z": z, "p": p,
                           "worse": bool(a.mean() > base.mean() and p < 0.05)}
        print(f"  {name:18} mean {a.mean():.3f}  z={z:+.2f} p={p:.2g} "
              f"{'WORSE' if out['kl'][name]['worse'] else ''}")
    out["kl"]["pred_0p"] = {"mean": base.mean(), "median": float(np.median(base))}
    out["kl"]["_sanity_true_diff"] = {"mean": float(np.mean(kl["_sanity_true_diff"]))}

    # --- best-of-16
    rows = json.loads((WORK / "eval16_pairs.json").read_text())
    preds = np.load(WORK / "preds_eval16_armA.npy").astype(np.float64)
    resid = np.load(WORK / "resid_holdout.npy").astype(np.float64)
    ho = np.load(WORK / "resid_holdout_rows.npy")
    ho_row = {int(t): k for k, t in enumerate(ho)}
    by_pos = {}
    for k, r in enumerate(rows):
        if r["tidx"] in ho_row:
            by_pos.setdefault(r["tidx"], []).append(k)
    pos = sorted(p for p in by_pos if len(by_pos[p]) >= 8)
    cos_first, cos_meanS, cos_best = [], [], []
    for p in pos:
        ks = by_pos[p]
        c = cosine(preds[ks], resid[ho_row[p]][None, :])
        cos_first.append(c[0]); cos_meanS.append(c.mean()); cos_best.append(c.max())
    rng = np.random.default_rng(0)
    null_best = []
    for _ in range(20):
        perm = rng.permutation(len(pos))
        nb = []
        for j, p in enumerate(pos):
            q = pos[perm[j]]
            if q == p:
                continue
            c = cosine(preds[by_pos[p]], resid[ho_row[q]][None, :])
            nb.append(c.max())
        null_best.append(np.mean(nb))
    out["bo16"] = {
        "n_pos": len(pos),
        "single_sample": float(np.mean(cos_first)),
        "mean_of_16": float(np.mean(cos_meanS)),
        "best_of_16_oracle": float(np.mean(cos_best)),
        "best_of_16_permuted_null": float(np.mean(null_best)),
    }
    print(f"[bo16] n={len(pos)}  single {out['bo16']['single_sample']:.3f}  "
          f"best-of-16 {out['bo16']['best_of_16_oracle']:.3f}  "
          f"permuted-null {out['bo16']['best_of_16_permuted_null']:.3f}")

    (WORK / "kl_bo16_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
