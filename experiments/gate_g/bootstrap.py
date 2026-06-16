"""Gate G Stage 0 significance: doc-clustered paired bootstrap.

Loads per-position eval cos for each arm (ckpt2_<arm>/eval_cos.npy +
eval_tidx.npy), aligns on the common holdout positions by tidx, clusters by
doc_idx (meta2), and bootstraps the paired mean differences:

  text - none  (value-add beyond h_20)
  text - cat   (beats the raw-context control)
  text - s20   (diff-specificity, if the s20 arm is present)

10k resamples over DOCS (not positions) -> honest CI given 6 positions/doc.
"""
import json
import sys
from pathlib import Path
import numpy as np

WORK = Path(__file__).parent
GC = WORK.parent / "gate_c"
meta = json.loads((GC / "meta2.json").read_text())
doc_of = {i: m["doc_idx"] for i, m in enumerate(meta)}

import os
PREF = os.environ.get("CKPT_PREFIX", "ckpt2_")
ARMS = sys.argv[1:] or ["none", "cat", "text"]
cos, tidx = {}, {}
for a in ARMS:
    d = WORK / f"{PREF}{a}"
    cos[a] = np.load(d / "eval_cos.npy")
    tidx[a] = np.load(d / "eval_tidx.npy")

# align on common tidx
common = set(tidx[ARMS[0]].tolist())
for a in ARMS[1:]:
    common &= set(tidx[a].tolist())
common = sorted(common)
print(f"arms={ARMS}  common holdout positions: {len(common)}")
idx = {a: {int(t): k for k, t in enumerate(tidx[a])} for a in ARMS}
C = {a: np.array([cos[a][idx[a][t]] for t in common]) for a in ARMS}
docs = np.array([doc_of[t] for t in common])
uniq = np.unique(docs)
# group position-indices by doc for fast resampling
by_doc = {d: np.where(docs == d)[0] for d in uniq}

print("\nmean cos per arm:")
for a in ARMS:
    print(f"  {a:5s} {C[a].mean():.4f}")

rng = np.random.default_rng(0)
B = 10000


def boot(diff):
    obs = diff.mean()
    stats = np.empty(B)
    for b in range(B):
        ds = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([by_doc[d] for d in ds])
        stats[b] = diff[sel].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())  # two-sided
    return obs, lo, hi, p, (diff > 0).mean()


print(f"\ndoc-clustered paired bootstrap ({len(uniq)} docs, B={B}):")
base = "text" if "text" in ARMS else ARMS[-1]
for other in [a for a in ARMS if a != base]:
    diff = C[base] - C[other]
    obs, lo, hi, p, winfrac = boot(diff)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {base} - {other:5s}: {obs:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
          f"p={p:.4f} {star}  wins {winfrac:.1%} of positions")
