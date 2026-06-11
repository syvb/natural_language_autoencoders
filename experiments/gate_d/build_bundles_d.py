"""Gate D1: evidence bundles + templated labels (arm T).

Bundles (labeler input, one per position): context tail, target token,
dominant heads with attended sources (token + short context + weight +
head's share of the write), total write norm percentile ("quiet" flag).
Logit-lens tokens are EXCLUDED from bundles — inspection showed they are
mostly junk for mid-stack attention writes; they stay in attn_evidence for
the groundedness union only.

Arm T: deterministic rendering of the same bundle (perfectly grounded,
rigid register).

Outputs: bundles_d.json {i: bundle}, armT_{train,eval}.json [{text,tidx}],
quiet_stats printed.
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
QUIET_PCTL = 10

meta = json.loads((WORK / "meta_d.json").read_text())
ev = {}
for line in (WORK / "attn_evidence.jsonl").open():
    r = json.loads(line)
    ev[r["i"]] = r
norms = np.array([ev[i]["total_norm"] for i in range(len(meta))])
quiet_thr = np.percentile(norms, QUIET_PCTL)

bundles = {}
armT = {}
for i, m in enumerate(meta):
    r = ev[i]
    heads = []
    for h in r["heads"][:6]:
        srcs = [{"token": s["tok"], "near": s["ctx"], "weight": s["w"]}
                for s in h["src"][:4] if s["w"] >= 0.03]
        if srcs:
            heads.append({"share_of_write": h["frac"], "attends_to": srcs})
    bundles[str(i)] = {
        "context_tail": m["context_tail"],
        "target_token": m["token_at_pos"],
        "write_norm": r["total_norm"],
        "quiet": bool(r["total_norm"] < quiet_thr),
        "heads": heads,
    }
    # templated rendering
    if r["total_norm"] < quiet_thr:
        t = (f"The attention write at '{m['token_at_pos'].strip()}' is weak; "
             f"no single retrieval dominates.")
    else:
        parts = []
        for h in heads[:4]:
            ss = ", ".join(f"'{s['token'].strip()}' (in \"{s['near'].strip()}\")"
                           for s in h["attends_to"][:2])
            parts.append(f"a head carrying {h['share_of_write']:.0%} of the "
                         f"write retrieves {ss}")
        t = (f"At the token '{m['token_at_pos'].strip()}', attention retrieves "
             f"prior context: " + "; ".join(parts) + ".")
    armT[i] = t

(WORK / "bundles_d.json").write_text(json.dumps(bundles))
train = [{"text": armT[i], "tidx": i} for i, m in enumerate(meta)
         if m["split"] == "train"]
hold = [{"text": armT[i], "tidx": i} for i, m in enumerate(meta)
        if m["split"] == "holdout"]
(WORK / "armT_train.json").write_text(json.dumps(train))
(WORK / "armT_eval.json").write_text(json.dumps(hold))
nq = sum(1 for b in bundles.values() if b["quiet"])
print(f"bundles: {len(bundles)} ({nq} quiet, thr={quiet_thr:.2f}); "
      f"armT train {len(train)} eval {len(hold)}")
print("--- sample armT:", armT[100])
