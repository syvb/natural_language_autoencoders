"""Gate D0.3 scoring: are zero-shot AV reads of attn writes grounded in the
attention evidence? (pre-registered gate)

Per read, content words = lowercased alphabetic tokens len>=3 minus
stopwords. Two scores:
  full : fraction of read content words found in {attended source contexts
         (all dominant heads) UNION local context tail UNION logit-lens top}
  src  : fraction found in attended-source-context words MINUS the local
         context-tail words (source-specific clause — paraphrasing nearby
         text cannot score here)
Null: same reads scored against shuffled positions' evidence (20 perms).
GO requires real > null with doc-clustered bootstrap CI > 0 for BOTH.

Usage: python3 ground_score.py [job=attn_read]
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
WORD = re.compile(r"[a-zA-Z]{3,}")
STOP = set("""the and for that this with from have has was were are been being
not but they them their there here when where which while what who whom whose
its his her she him you your our out into onto over under about above below
between because before after during against further then once all any both
each few more most other some such only own same than too very can will just
should now also like one two may might must shall does did doing would could
text token tokens word words sentence passage context phrase mention mentions
describes discussing discusses describing focuses focusing appears refers
referring related concerning regarding involving content topic subject
specific particular likely possibly perhaps generally currently
activation activations final""".split())


def words(s):
    return {w.lower() for w in WORD.findall(s or "")} - STOP


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "attn_read"
    meta = json.loads((WORK / "meta_d.json").read_text())
    ev = {}
    for line in (WORK / "attn_evidence.jsonl").open():
        r = json.loads(line)
        src_w = set()
        for h in r["heads"]:
            for s in h["src"]:
                src_w |= words(s["ctx"])
        ev[r["i"]] = {"src": src_w, "lens": words(" ".join(r["lens_top"]))}
    tail_w = {i: words(m["context_tail"]) for i, m in enumerate(meta)}

    reads = {}
    for fn in glob.glob(str(WORK / "gend_out" / f"{job}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            t = r["explanation"] if r["explanation"] is not None else r["raw"]
            if t and len(t) > 20:
                reads.setdefault(r["i"], []).append(t)
    idx = sorted(set(reads) & set(ev))
    print(f"[ground] {len(idx)} positions with reads")

    def score_against(i_read, i_ev):
        rw = set().union(*(words(t) for t in reads[i_read]))
        if not rw:
            return None, None
        full_set = ev[i_ev]["src"] | tail_w[i_ev] | ev[i_ev]["lens"]
        src_set = ev[i_ev]["src"] - tail_w[i_ev]
        return (len(rw & full_set) / len(rw), len(rw & src_set) / len(rw))

    real_f, real_s, docs = [], [], []
    for i in idx:
        f, s = score_against(i, i)
        if f is None:
            continue
        real_f.append(f); real_s.append(s); docs.append(meta[i]["doc_idx"])
    real_f, real_s, docs = map(np.array, (real_f, real_s, docs))

    rng = np.random.default_rng(0)
    null_f, null_s = [], []
    for _ in range(20):
        perm = rng.permutation(len(idx))
        nf, ns = [], []
        for j, i in enumerate(idx):
            q = idx[perm[j]]
            if q == i:
                continue
            f, s = score_against(i, q)
            if f is not None:
                nf.append(f); ns.append(s)
        null_f.append(np.mean(nf)); null_s.append(np.mean(ns))
    null_f, null_s = np.mean(null_f), np.mean(null_s)

    # doc-clustered bootstrap on (real − null) margins
    uniq = np.unique(docs)
    res = {}
    for name, vals, null in [("full", real_f, null_f), ("src", real_s, null_s)]:
        sums = np.zeros(len(uniq)); cnts = np.zeros(len(uniq))
        k = np.searchsorted(uniq, docs)
        np.add.at(sums, k, vals); np.add.at(cnts, k, 1)
        picks = rng.integers(0, len(uniq), size=(10000, len(uniq)))
        boots = sums[picks].sum(1) / cnts[picks].sum(1) - null
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[name] = {"real": float(vals.mean()), "null": float(null),
                     "margin": float(vals.mean() - null),
                     "margin_ci": [float(lo), float(hi)]}
        print(f"[ground] {name}: real {vals.mean():.4f}  null {null:.4f}  "
              f"margin {vals.mean()-null:+.4f} CI [{lo:+.4f},{hi:+.4f}]")
    res["gate"] = ("GO" if all(res[k]["margin_ci"][0] > 0 for k in ("full", "src"))
                   else "SOFT-FAIL")
    print(f"[ground] -> {res['gate']}")
    (WORK / f"ground_{job}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
