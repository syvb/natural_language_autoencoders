"""Gate E1 scoring: edit-span hit rate in zero-shot delta reads.

A read scores a target-hit if it contains the edit's target span (the
NEW content — only knowable from the delta) and a source-hit for the
original span. Matching: case-insensitive substring on the span stripped
of leading/trailing punctuation; spans shorter than 4 chars are skipped.
Null: reads scored against shuffled positions' spans (20 perms).
Pre-registered GO: target-hit >= 5% absolute AND >= 3x null (doc-clustered
bootstrap CI > 0). Also reports by edit type and by dist_from_edit tercile.
"""

import glob
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent


def norm_span(s):
    return s.strip().strip('.,;:"\'!?()').lower()


def main():
    meta = json.loads((WORK / "meta_e.json").read_text())
    reads = {}
    for fn in glob.glob(str(WORK / "gene_out" / "delta_read*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            t = r["explanation"] if r["explanation"] is not None else r["raw"]
            if t:
                reads.setdefault(r["i"], []).append(t.lower())
    idx = [i for i in sorted(reads)
           if len(norm_span(meta[i]["target"])) >= 4]
    print(f"[span] {len(idx)} scored positions")

    def hit(i_read, i_span):
        tgt = norm_span(meta[i_span]["target"])
        src = norm_span(meta[i_span]["source"])
        joined = " ".join(reads[i_read])
        return (tgt in joined, src in joined)

    th = np.array([hit(i, i)[0] for i in idx], dtype=float)
    sh = np.array([hit(i, i)[1] for i in idx], dtype=float)
    rng = np.random.default_rng(0)
    null_t = []
    for _ in range(20):
        perm = rng.permutation(len(idx))
        null_t.append(np.mean([hit(i, idx[perm[k]])[0]
                               for k, i in enumerate(idx) if idx[perm[k]] != i]))
    null_t = float(np.mean(null_t))

    docs = np.array([meta[i]["doc_idx"] for i in idx])
    uniq = np.unique(docs)
    k = np.searchsorted(uniq, docs)
    sums = np.zeros(len(uniq)); cnts = np.zeros(len(uniq))
    np.add.at(sums, k, th); np.add.at(cnts, k, 1)
    picks = rng.integers(0, len(uniq), size=(10000, len(uniq)))
    boots = sums[picks].sum(1) / cnts[picks].sum(1) - null_t
    lo, hi = np.percentile(boots, [2.5, 97.5])

    res = {"n": len(idx), "target_hit": float(th.mean()),
           "source_hit": float(sh.mean()), "null_target": null_t,
           "margin_ci": [float(lo), float(hi)],
           "gate": ("GO" if th.mean() >= 0.05 and th.mean() >= 3 * null_t
                    and lo > 0 else "FAIL")}
    print(f"[span] target-hit {th.mean():.4f} (null {null_t:.4f}) "
          f"source-hit {sh.mean():.4f} margin CI [{lo:+.4f},{hi:+.4f}] "
          f"-> {res['gate']}")
    by = {}
    for grp, key in [("type", lambda m: m["type"]),
                     ("dist", lambda m: ("near" if m["dist_from_edit"] < 10
                                         else "mid" if m["dist_from_edit"] < 40
                                         else "far"))]:
        agg = {}
        for j, i in enumerate(idx):
            agg.setdefault(key(meta[i]), []).append(th[j])
        by[grp] = {g: [float(np.mean(v)), len(v)] for g, v in agg.items()}
        print(f"[span] by {grp}:", {g: f"{m:.3f}(n={n})"
                                    for g, (m, n) in by[grp].items()})
    res["by"] = by
    (WORK / "span_score_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
