"""Propensity permutation null for the order-only result.

Objection being tested: a model with NO ordering mechanism still shows "dominant
concept appears earlier" if its per-line propensity to mention a concept rises with
that concept's injected magnitude — first-mention order inherits dose-response
mechanically. Null: within each both-present row, shuffle line order (keeping each
line's (yellow, syco) label pair intact) and recompute P(yellow first); the null
slope is the propensity-only component, the observed-minus-null excess is genuine
position allocation.

Result (2026-07-04): v3 lines obs +0.88 / null +0.45 (excess +0.43); kitft chunks
obs +0.17 / null +0.46 (excess −0.30); kitft lines obs +0.10 / null +0.31 (−0.21).
~Half of v3's order slope is propensity; the other half is real ordering. kitft is
BELOW its own null — its layout is template-locked (register/quote/next-token slots),
so random placement of its own mentions would track the ratio better.
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NSHUF = 400


def first_idx(l):
    for i, x in enumerate(l):
        if x == 2:
            return i + 1
    return None


def yf(fy, fs):
    if fs is None:
        return 1.0
    if fy is None:
        return 0.0
    return 1.0 if fy < fs else 0.0 if fs < fy else 0.5


def slope(pts):
    X = np.stack([np.ones(len(pts)), np.array([p[0] for p in pts])], 1)
    y = np.array([p[1] for p in pts])
    w = np.zeros(2)
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = -(X * (p * (1 - p))[:, None]).T @ X - 1e-6 * np.eye(2)
        s = np.linalg.solve(H, g)
        w -= s
        if np.abs(s).max() < 1e-8:
            break
    return w[1]


rng = np.random.default_rng(0)
for name, fn in (("v3 (lines)", "ratio_judged_v3.json"),
                 ("kitft (chunks)", "ratio_judged_kitft_chunks.json"),
                 ("kitft (lines)", "ratio_judged_kitft.json")):
    obs_pts, null_pts = [], []
    perlam_o, perlam_n = defaultdict(list), defaultdict(list)
    for x in json.load(open(f"{HERE}/results/{fn}")):
        fy, fs = first_idx(x["yellow"]), first_idx(x["syco"])
        if fy is None or fs is None:
            continue  # both-present rows only
        obs = yf(fy, fs)
        pairs = list(zip(x["yellow"], x["syco"]))
        vals = []
        for _ in range(NSHUF):
            rng.shuffle(pairs)
            vals.append(yf(first_idx([p[0] for p in pairs]), first_idx([p[1] for p in pairs])))
        null = float(np.mean(vals))
        obs_pts.append((x["lam"], obs)); null_pts.append((x["lam"], null))
        perlam_o[x["lam"]].append(obs); perlam_n[x["lam"]].append(null)
    so, sn = slope(obs_pts), slope(null_pts)
    print(f"{name:15s} n={len(obs_pts):4d}  observed slope {so:+.3f}  propensity-null slope {sn:+.3f}  excess {so-sn:+.3f}")
    for l in sorted(perlam_o):
        if len(perlam_o[l]) >= 15:
            print(f"    lam={l:+4.1f}: obs {np.mean(perlam_o[l]):.3f}  null {np.mean(perlam_n[l]):.3f}  (n={len(perlam_o[l])})")
print("PERMUTATION_NULL_DONE")
