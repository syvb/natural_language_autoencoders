"""Assemble Gate C arm datasets from session-1 outputs (runs locally).

Inputs:  meta2.json, gen2_out/*.jsonl, acts18/22.npy (for targets),
         later hybrid_labels.json (arm B).
Outputs: diff_targets.npy (raw v22-v18, fp32)
         hybrid_inputs.json (for hybrid_v2.py submit)
         arm pair files: arm0_{train,eval}.json (L20 expl),
         arm0p_{train,eval}.json (L18+L22 concat), armA_{train,eval}.json
         (diff1 text), armB_train.json + armB-input eval (diff1-as-eval-input
         is shared; arm B eval uses hybrid labels on holdout — built only if
         hybrid labels for holdout exist, else arm B evaluates on its own
         register via train-split heldback docs).
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
CJK = re.compile(r"[぀-ヿ一-鿿]")
MATCHED_N = 8000

meta = json.loads((WORK / "meta2.json").read_text())
n = len(meta)

def load_job(name):
    rows = {}
    p = WORK / "gen2_out" / f"{name}.jsonl"
    for line in p.open():
        r = json.loads(line)
        rows[(r["i"], r["s"])] = r
    return rows

def text_of(r):
    t = r["explanation"] if r["explanation"] is not None else r["raw"]
    return (t or "").strip()

def clean(t):  # filter: drop CJK/garbled per generation review
    return t and not CJK.search(t) and len(t) > 40

cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

if cmd in ("targets", "all"):
    a18 = np.load(WORK / "acts18.npy", mmap_mode="r")
    a22 = np.load(WORK / "acts22.npy", mmap_mode="r")
    np.save(WORK / "diff_targets.npy",
            (np.asarray(a22, dtype=np.float32) - np.asarray(a18, dtype=np.float32)))
    print("targets saved", a18.shape)

if cmd in ("hybrid_inputs", "all"):
    e18, e22 = load_job("expl18"), load_job("expl22")
    reads = load_job("hybrid_reads")
    hybrid_pos = sorted({i for (i, s) in reads})
    inputs = {}
    for i in hybrid_pos:
        rds = [text_of(reads[(i, s)]) for s in range(4) if (i, s) in reads]
        rds = [t for t in rds if t]
        if len(rds) < 3 or (i, 0) not in e18 or (i, 0) not in e22:
            continue
        inputs[str(i)] = {"source": meta[i]["context_tail"],
                          "token": meta[i]["token_at_pos"],
                          "expl_a": text_of(e18[(i, 0)]),
                          "expl_b": text_of(e22[(i, 0)]),
                          "diff_reads": rds}
    ev = load_job("eval16")
    hold_pos = sorted({i for (i, s) in ev})
    n_hold_added = 0
    for i in hold_pos:
        rds = [text_of(ev[(i, s)]) for s in range(4) if (i, s) in ev]
        rds = [t2 for t2 in rds if t2]
        if len(rds) < 3 or (i, 0) not in e18 or (i, 0) not in e22:
            continue
        inputs[str(i)] = {"source": meta[i]["context_tail"],
                          "token": meta[i]["token_at_pos"],
                          "expl_a": text_of(e18[(i, 0)]),
                          "expl_b": text_of(e22[(i, 0)]),
                          "diff_reads": rds}
        n_hold_added += 1
    (WORK / "hybrid_inputs.json").write_text(json.dumps(inputs))
    print(f"hybrid inputs: {len(inputs)} (incl. {n_hold_added} holdout)")

if cmd in ("arms", "all"):
    e18, e22 = load_job("expl18"), load_job("expl22")
    d1 = load_job("diff1")
    train = [i for i, m in enumerate(meta) if m["split"] == "train"]
    hold = [i for i, m in enumerate(meta) if m["split"] == "holdout"]

    def pairs(idx, maker):
        out = []
        for i in idx:
            t = maker(i)
            if t and clean(t):
                out.append({"text": t, "tidx": i})
        return out

    mk0p = lambda i: (f"Earlier: {text_of(e18[(i,0)])}\n\nLater: {text_of(e22[(i,0)])}"
                      if (i, 0) in e18 and (i, 0) in e22 else None)
    mkA = lambda i: text_of(d1[(i, 0)]) if (i, 0) in d1 else None

    for name, mk in [("arm0p", mk0p), ("armA", mkA)]:
        tr = pairs(train, mk)[:MATCHED_N]
        ev = pairs(hold, mk)
        (WORK / f"{name}_train.json").write_text(json.dumps(tr))
        (WORK / f"{name}_eval.json").write_text(json.dumps(ev))
        print(f"{name}: train {len(tr)}, eval {len(ev)}")
    # full-n slope run for arm A
    trA_full = pairs(train, mkA)
    (WORK / "armA_full_train.json").write_text(json.dumps(trA_full))
    print(f"armA_full: {len(trA_full)}")

if cmd in ("armB", "all"):
    p = WORK / "hybrid_labels.json"
    if p.exists():
        labels = json.loads(p.read_text())
        tr = [{"text": r["difference"], "tidx": int(i)}
              for i, r in labels.items()
              if meta[int(i)]["split"] == "train"
              and r.get("difference") and clean(r["difference"])][:MATCHED_N]
        (WORK / "armB_train.json").write_text(json.dumps(tr))
        ev = [{"text": r["difference"], "tidx": int(i)}
              for i, r in labels.items()
              if meta[int(i)]["split"] == "holdout"
              and r.get("difference") and clean(r["difference"])]
        (WORK / "armB_eval.json").write_text(json.dumps(ev))
        print(f"armB: train {len(tr)}, eval {len(ev)}")
    else:
        print("armB: hybrid_labels.json not present yet")
