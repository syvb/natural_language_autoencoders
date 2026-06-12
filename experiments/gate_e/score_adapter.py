"""Gate E2 scoring (pre-registered).

Per seed and per job:
  *_fwd : expected span = target (the NEW content)
  *_rev : expected span = source (reversed delta -> reversed edit)
Metrics: span-hit rate (substring, normalized) + shuffled-position null;
antisymmetry = among rr_fwd hits, fraction whose rev read hits source.
PASS (held-out type rr): fwd hit >= 25% (null <= 2%) AND antisym >= 50%.
MARGINAL: 10-25%. Also reports lex (in-distribution) numbers.
"""

import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent


def norm_span(s):
    return s.strip().strip('.,;:"\'!?()').lower()


def load(tag, job):
    out = {}
    p = WORK / "gene_out" / f"{tag}_{job}.jsonl"
    if not p.exists():
        return out
    for line in p.open():
        r = json.loads(line)
        t = r["explanation"] if r["explanation"] is not None else r["raw"]
        if t:
            out[r["i"]] = t.lower()
    return out


def main():
    meta = json.loads((WORK / "meta_e.json").read_text())
    rng = np.random.default_rng(0)
    results = {}
    for tag in ["W_seed0", "W_seed1"]:
        results[tag] = {}
        for job, field in [("rr_fwd", "target"), ("rr_rev", "source"),
                           ("lex_fwd", "target"), ("lex_rev", "source")]:
            reads = load(tag, job)
            idx = [i for i in sorted(reads)
                   if len(norm_span(meta[i][field])) >= 4]
            if not idx:
                continue
            hits = np.array([norm_span(meta[i][field]) in reads[i]
                             for i in idx], dtype=float)
            perm = rng.permutation(len(idx))
            null = np.mean([norm_span(meta[idx[perm[k]]][field]) in reads[i]
                            for k, i in enumerate(idx) if idx[perm[k]] != i])
            results[tag][job] = {"n": len(idx), "hit": float(hits.mean()),
                                 "null": float(null)}
            print(f"{tag} {job:8} n={len(idx):5} hit {hits.mean():.4f} "
                  f"null {null:.4f}")
        # antisymmetry on rr
        f, r = load(tag, "rr_fwd"), load(tag, "rr_rev")
        fwd_hit = [i for i in f if i in r
                   and len(norm_span(meta[i]["target"])) >= 4
                   and norm_span(meta[i]["target"]) in f[i]]
        anti = (np.mean([norm_span(meta[i]["source"]) in r[i]
                         for i in fwd_hit]) if fwd_hit else float("nan"))
        results[tag]["antisym"] = {"n_fwd_hits": len(fwd_hit),
                                   "rate": None if not fwd_hit else float(anti)}
        print(f"{tag} antisym: {anti if fwd_hit else 'n/a'} over "
              f"{len(fwd_hit)} fwd hits")
    (WORK / "e2_results.json").write_text(json.dumps(results, indent=2))
    rr = results.get("W_seed0", {}).get("rr_fwd", {})
    h = rr.get("hit", 0)
    verdict = ("PASS" if h >= 0.25 else "MARGINAL" if h >= 0.10 else "FAIL")
    print(f"held-out verdict (seed0 rr_fwd vs pre-registered bars): {verdict}")


if __name__ == "__main__":
    main()
