"""Isolate the REAL hallucinations by independent-consensus + strict extraction.

The single faithfulness judge over-flags (blind check: 7 faithful items called
hallucinated, 0 the reverse). To get a high-precision set, re-judge EVERY
currently-hallucinated item (nex CON/FAB) with a demanding extraction prompt on
TWO independent models — nex-n2-mini and openai/gpt-4o-mini — that must quote the
specific fabricated fact and rule out plausible-prediction / vague / meta /
misquote. A "confirmed" hallucination is one BOTH models independently flag.

Then measure within-position marginal FVE for the confirmed set (vs SUPPORTED),
with a permutation test, and compare to the loose category.

    JUDGE_WORKERS=150 python judge_strict.py
"""
import hashlib
import json
import os
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

import analyze

HERE, SE = analyze.HERE, analyze.SE
KEY, URL = analyze.KEY, analyze.URL
H = {"CONTRADICTED", "FABRICATED"}
MODELS = ["nex-agi/nex-n2-mini", "openai/gpt-4o-mini"]
PROMPT_VERSION = "strict-halluc-v1"
VERD = re.compile(r"<verdict>\s*(HALLUCINATION|NOT_HALLUCINATION)\s*</verdict>", re.I)

PROMPT = """You are auditing an interpretability note about a language model that was reading a passage. The note may correctly describe what the model is INTERNALLY predicting it will generate next — that is NOT a hallucination even if the content is not written in the passage yet.

Decide if the note contains a REAL hallucination: a SPECIFIC claim that
  (a) states a concrete fact, named entity, quotation, number, or event as being present in / true of the passage, AND
  (b) does not actually appear in the passage or its true continuation, AND
  (c) is not a reasonable prediction of what the model is about to generate next, AND
  (d) is not merely a description of genre, tone, structure, topic, or a hedged/vague guess.

Passage:
<passage>
{passage}
</passage>
True continuation:
<continuation>
{continuation}
</continuation>
Note:
<note>
{note}
</note>

Think briefly. If the note contains a real hallucination, quote the single most clearly fabricated claim verbatim in <claim>...</claim>, then answer <verdict>HALLUCINATION</verdict>. Otherwise (faithful, a plausible prediction, vague/hedged, or pure genre/tone commentary) answer <verdict>NOT_HALLUCINATION</verdict>."""

caches = {m: (HERE / "results" / f".strict_cache_{re.sub(r'[^a-z0-9]+','-',m.lower())}.json") for m in MODELS}
cache = {m: (json.load(open(p)) if p.exists() else {}) for m, p in caches.items()}


def ask(args):
    model, prompt = args
    k = hashlib.sha1((PROMPT_VERSION + "|" + model + "|" + prompt).encode()).hexdigest()
    c = cache[model]
    if k in c and c[k] is not None:
        return model, k, c[k]
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            m = VERD.search(r["choices"][0]["message"]["content"])
            if m:
                return model, k, m.group(1).upper()
        except Exception:
            continue
    return model, k, None


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def ols(rows, is_target):
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); f = np.array(is_target, float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), se, float(2 * stats.t.sf(abs(b[-1] / se), dof))


def main():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))

    items = {}
    for arm, fn in [("mat", "subset_scores_mat.json"), ("std", "subset_scores_std.json")]:
        items[arm] = []
        for e in json.load(open(HERE / "results" / fn))["entries"]:
            for ri, rec in enumerate(e["rollouts"]):
                if not rec:
                    continue
                m = marg(rec)
                for k in range(len(m)):
                    key = f"{arm}|{e['ci']}|{ri}|{k}"
                    v = faith.get(key)
                    if v is not None:
                        items[arm].append(dict(key=key, ci=e["ci"], pos=k, mg=m[k], v=v, text=rec["units"][k]))

    # jobs: strict prompt on both models, for every currently-hallucinated item
    jobs, meta = [], []
    for arm in ("mat", "std"):
        for it in items[arm]:
            if it["v"] in H:
                pr = PROMPT.format(passage=man[it["ci"]]["prefix_text"],
                                   continuation=man[it["ci"]]["answer_text"], note=it["text"])
                for model in MODELS:
                    jobs.append((model, pr)); meta.append((model, it["key"]))
    print(f"[judge] {len(jobs)} strict calls ({len(jobs)//len(MODELS)} items x {len(MODELS)} models)…", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "150"))) as ex:
        buf = []
        for c0 in range(0, len(jobs), 500):
            buf += list(ex.map(ask, jobs[c0:c0 + 500]))
            for model, k, v in buf[c0:]:
                cache[model][k] = v
            for m, p in caches.items():
                json.dump(cache[m], open(p, "w"))
            print(f"  {min(c0 + 500, len(jobs))}/{len(jobs)}", flush=True)
    for (model, key), (_, _, v) in zip(meta, buf):
        res[(model, key)] = v
    fails = sum(1 for v in res.values() if v is None)
    print(f"[judge] done, {fails} unparsed", flush=True)

    # verdict per model + consensus
    strict = {}   # key -> dict(model->bool)
    for arm in ("mat", "std"):
        for it in items[arm]:
            if it["v"] in H:
                strict[it["key"]] = {m: (res.get((m, it["key"])) == "HALLUCINATION") for m in MODELS}
    json.dump({k: v for k, v in strict.items()}, open(HERE / "results" / "strict_halluc.json", "w"))

    out = {}
    print("\n" + "=" * 74)
    rng = random.Random(0)
    for arm, label, fn in [("mat", "matryoshka (lines)", None), ("std", "standard (sentences)", None)]:
        rows = items[arm]
        sup = [(it["pos"], it["mg"]) for it in rows if it["v"] == "SUPPORTED"]
        loose = [it for it in rows if it["v"] in H]
        n_nex = sum(strict[it["key"]][MODELS[0]] for it in loose)
        n_gpt = sum(strict[it["key"]][MODELS[1]] for it in loose)
        both = [it for it in loose if all(strict[it["key"]].values())]
        either = [it for it in loose if any(strict[it["key"]].values())]
        # inter-model agreement on the loose set
        agree = np.mean([strict[it["key"]][MODELS[0]] == strict[it["key"]][MODELS[1]] for it in loose])
        print(f"\n{label}: loose-hallucinated n={len(loose)}")
        print(f"  strict-HALLUCINATION kept: nex {n_nex} ({n_nex/len(loose):.0%}) | gpt-4o-mini {n_gpt} ({n_gpt/len(loose):.0%}) | BOTH {len(both)} ({len(both)/len(loose):.0%})")
        print(f"  inter-model agreement on loose set: {agree:.0%}")

        def within(group):
            rr = [(p, m) for p, m in sup] + [(it["pos"], it["mg"]) for it in group]
            tgt = [0] * len(sup) + [1] * len(group)
            b, se, p = ols(rr, tgt)
            return b, p
        for nm, grp in [("loose (all CON/FAB)", loose), ("strict: nex only", [it for it in loose if strict[it['key']][MODELS[0]]]),
                        ("strict: CONFIRMED (both)", both)]:
            b, p = within(grp)
            print(f"  within-pos vs SUPPORTED — {nm:26s} coef {b:+.4f} p={p:.2e}  (n={len(grp)})")
        # permutation: is CONFIRMED better than a random same-size subset of loose?
        real_b, _ = within(both)
        keys = [it for it in loose]
        perm = []
        for _ in range(1000):
            idx = list(range(len(loose))); rng.shuffle(idx)
            samp = [loose[i] for i in idx[:len(both)]]
            perm.append(within(samp)[0])
        perm = np.array(perm); pperm = np.mean(np.abs(perm) >= abs(real_b))
        print(f"  permutation (CONFIRMED vs random same-size subset of loose): coef {real_b:+.4f}, p_perm={pperm:.3f}")
        out[arm] = dict(n_loose=len(loose), n_confirmed=len(both), frac_confirmed=len(both)/len(loose),
                        inter_model_agree=float(agree),
                        coef_loose=within(loose), coef_confirmed=[real_b, pperm])
    json.dump(out, open(HERE / "results" / "strict_stats.json", "w"), indent=1)
    print("\n[saved] results/strict_halluc.json, strict_stats.json")


if __name__ == "__main__":
    main()
