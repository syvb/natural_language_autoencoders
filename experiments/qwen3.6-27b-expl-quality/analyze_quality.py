"""Aggregate the judge's output into calibrated, clustered statistics.

Reads results/quality_<judge-slug>.json (from judge_quality.py) and writes
results/quality_analysis_<slug>.json plus a console report.

Design choices (from the pre-registration review):
  * hallucination is reported under FOUR denominators — per checkable claim,
    count per explanation, per 100 words of explanation text, and
    P(explanation contains >=1) — because judge-side claim segmentation makes
    any single denominator format-sensitive. Claims-per-100-words per arm is
    printed as the segmentation-parity diagnostic.
  * all cross-arm differences are per-context PAIRED diffs with a cluster
    bootstrap over contexts (claims cluster within explanations).
  * the skyline / derangement-floor calibration arms bound the judge's
    false-positive floor and sensitivity ceiling; real-arm rates are also
    reported rescaled to that range.
  * paired probe: both A/B orders per context -> win/tie/loss per dimension,
    position-consistent wins, and the order-flip rate.

    python analyze_quality.py                 # default nex judge
    python analyze_quality.py --judge openai/gpt-4o-mini
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HALLUC = {"UNSUPPORTED", "CONTRADICTED"}
CHECKABLE = {"SUPPORTED", "PREDICTIVE", "UNSUPPORTED", "CONTRADICTED"}
SCORES = ("coherence", "usefulness", "informativeness")
PDIMS = ("more_useful", "more_grounded", "more_coherent", "overall")
B = 10_000


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def boot_ci(vals, rng):
    """mean + 95% cluster-bootstrap CI over contexts."""
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    idx = rng.integers(0, len(a), (B, len(a)))
    means = a[idx].mean(1)
    return {"mean": float(a.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "n": int(len(a))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="nex-agi/nex-n2-mini")
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest_quality.json"))
    args = ap.parse_args()
    slug = re.sub(r"[^a-z0-9]+", "-", args.judge.lower()).strip("-")
    rep = json.load(open(HERE / "results" / f"quality_{slug}.json"))
    man = {c["ci"]: c for c in json.load(open(args.manifest))["contexts"]}
    rng = np.random.default_rng(0)

    # per-context records per leg
    legs = {}
    for r in rep["absolute"]:
        legs.setdefault(r["arm"], {})[r["ci"]] = r
    arms = rep["meta"]["arms"]                      # the two real arms
    dom = {ci: man[ci]["domain"] for ci in man}
    expl_words = {}                                 # (leg, ci) -> word count
    for ef in [HERE / "results" / "explanations_mat.json",
               HERE / "results" / "explanations_std.json"]:
        d = json.load(open(ef))
        for e in d["entries"]:
            expl_words[(d["meta"]["model"], e["ci"])] = len(
                " ".join(e["explanations"][0]).split())

    def ctx_stats(leg, ci):
        r = legs[leg].get(ci)
        if r is None:
            return None
        lab = Counter(c["label"] for c in r["claims"])
        chk = sum(lab[l] for l in CHECKABLE)
        hal = sum(lab[l] for l in HALLUC)
        words = expl_words.get((leg, ci))
        if words is None and leg.startswith("floor_"):
            words = expl_words.get((leg.split("_", 1)[1], ci))
        return {"chk": chk, "hal": hal, "n_claims": len(r["claims"]),
                "labels": lab, "words": words,
                "rate": (hal / chk) if chk else np.nan,
                "per100w": (100 * hal / words) if words else np.nan,
                "any": float(hal > 0),
                **{s: r[s] for s in SCORES}}

    groups = [("all", None), ("pretrain", "pretrain"), ("wildchat", "wildchat")]
    out = {"meta": rep["meta"], "absolute": {}, "diffs": {}, "paired": {}}

    for gname, gdom in groups:
        cis = sorted(ci for ci in man if gdom is None or dom[ci] == gdom)
        gout = {}
        for leg in legs:
            rows = [s for ci in cis if (s := ctx_stats(leg, ci))]
            if not rows:
                continue
            lab_tot = Counter()
            for s in rows:
                lab_tot.update(s["labels"])
            tot_chk = sum(s["chk"] for s in rows)
            tot_hal = sum(s["hal"] for s in rows)
            gout[leg] = {
                "n": len(rows),
                "labels": dict(lab_tot),
                "claims_per_expl": float(np.mean([s["n_claims"] for s in rows])),
                "claims_per_100w": float(np.nanmean(
                    [100 * s["n_claims"] / s["words"] for s in rows if s["words"]])),
                "halluc_rate_pooled": tot_hal / max(1, tot_chk),
                "halluc_rate_ctx": boot_ci([s["rate"] for s in rows], rng),
                "halluc_count": boot_ci([s["hal"] for s in rows], rng),
                "halluc_per100w": boot_ci([s["per100w"] for s in rows], rng),
                "p_any_halluc": boot_ci([s["any"] for s in rows], rng),
                **{s2: boot_ci([s[s2] for s in rows], rng) for s2 in SCORES},
            }
        out["absolute"][gname] = gout

        # paired per-context diffs between the two real arms
        a1, a2 = arms
        dd = {}
        for metric in ("rate", "hal", "per100w", "any") + SCORES:
            diffs = []
            for ci in cis:
                s1, s2 = ctx_stats(a1, ci), ctx_stats(a2, ci)
                if s1 and s2:
                    v1 = s1[metric] if metric in s1 else np.nan
                    v2 = s2[metric] if metric in s2 else np.nan
                    diffs.append(v1 - v2)
            dd[f"{metric}_{a1}_minus_{a2}"] = boot_ci(diffs, rng)
        out["diffs"][gname] = dd

        # paired-probe stats (both orders)
        if "paired" in rep:
            rows = [r for r in rep["paired"]
                    if gdom is None or dom[r["ci"]] == gdom]
            by_ci = {}
            for r in rows:
                by_ci.setdefault(r["ci"], {})[r["order"]] = r
            pout = {}
            for dim in PDIMS:
                votes = Counter(r[dim] for r in rows)
                n_dec = votes[a1] + votes[a2]
                p, lo, hi = wilson(votes[a1], n_dec)
                both = [v for v in by_ci.values() if len(v) == 2]
                cons = Counter()
                flip = 0
                for v in both:
                    x, y = v["ms"][dim], v["sm"][dim]
                    if x == y != "TIE":
                        cons[x] += 1
                    elif x != y and "TIE" not in (x, y):
                        flip += 1
                pout[dim] = {
                    "votes": dict(votes), "n_decisive": n_dec,
                    f"win_{a1}": p, "win_lo": lo, "win_hi": hi,
                    "n_both_orders": len(both),
                    "consistent": dict(cons),
                    "flip_rate": flip / max(1, len(both)),
                }
            out["paired"][gname] = pout

    jout = HERE / "results" / f"quality_analysis_{slug}.json"
    jout.write_text(json.dumps(out, indent=1))
    print(f"[write] {jout}\n")

    # ── console report ───────────────────────────────────────────────────────
    def fmt(b, pct=False, pm=False):
        if b is None:
            return "—"
        m, lo, hi = b["mean"], b["lo"], b["hi"]
        if pct:
            return f"{m:.1%} [{lo:.1%},{hi:.1%}]"
        return f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]" if pm else f"{m:.2f} [{lo:.2f},{hi:.2f}]"

    for gname, _ in groups:
        gout = out["absolute"][gname]
        print(f"═══ {gname} ═══")
        for leg in sorted(gout, key=lambda l: (l not in arms, l)):
            g = gout[leg]
            print(f"  {leg:10s} n={g['n']:3d} | halluc/checkable {fmt(g['halluc_rate_ctx'], pct=True)} "
                  f"| per-100w {fmt(g['halluc_per100w'])} | P(>=1) {fmt(g['p_any_halluc'], pct=True)}")
            print(f"  {'':10s} claims/expl {g['claims_per_expl']:.1f} "
                  f"(per100w {g['claims_per_100w']:.1f}) | coh {fmt(g['coherence'])} "
                  f"| use {fmt(g['usefulness'])} | info {fmt(g['informativeness'])}")
        a1, a2 = arms
        print(f"  paired diffs ({a1}−{a2}): "
              f"halluc-rate {fmt(out['diffs'][gname][f'rate_{a1}_minus_{a2}'], pm=True)} | "
              f"use {fmt(out['diffs'][gname][f'usefulness_{a1}_minus_{a2}'], pm=True)} | "
              f"info {fmt(out['diffs'][gname][f'informativeness_{a1}_minus_{a2}'], pm=True)}")
        if gname in out["paired"]:
            for dim, p in out["paired"][gname].items():
                print(f"  paired {dim:14s}: {a1} wins {p[f'win_{a1}']:.0%} "
                      f"[{p['win_lo']:.0%},{p['win_hi']:.0%}] of {p['n_decisive']} decisive "
                      f"| flip {p['flip_rate']:.0%} | votes {p['votes']}")
        print()


if __name__ == "__main__":
    main()
