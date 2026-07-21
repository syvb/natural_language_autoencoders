"""Re-judge the currently-hallucinated items with the MODEL'S OWN continuation
added to the prompt. If a "hallucination" actually described what the model went
on to generate (an unverbalized prediction), the judge should now flip it to
SUPPORTED. Then measure how many flip, and re-run the within-position FVE on the
residual "real (even given the model's output)" hallucinations.

    JUDGE_WORKERS=150 python rejudge_withgen.py
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
KEY, URL, JUDGE = analyze.KEY, analyze.URL, analyze.JUDGE
H = {"CONTRADICTED", "FABRICATED"}
PROMPT_VERSION = "faith-withgen-v1"
CACHE = HERE / "results" / f".withgen_cache_{re.sub(r'[^a-z0-9]+','-',JUDGE.lower())}.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
ANS = re.compile(r"<answer>\s*(SUPPORTED|CONTRADICTED|FABRICATED|META)\s*</answer>", re.I)

PROMPT = """You are auditing an interpretability note about a language model reading a passage. The note describes the model's internal state / what it is about to generate. To help you, you are given BOTH the true corpus continuation AND several continuations the MODEL ITSELF actually generated from this exact point — the note is faithful if it matches what the model represents or generates, even if that differs from the corpus text.

Passage (model had read this much):
<passage>
{passage}
</passage>
True corpus continuation:
<corpus_next>
{continuation}
</corpus_next>
What the MODEL actually generated next (greedy + samples):
<model_next>
{modelgen}
</model_next>

The note:
<note>
{note}
</note>

Classify the note:
- SUPPORTED: consistent with the passage, the corpus continuation, OR what the model actually generated (a correct read of the model's own direction counts as supported).
- CONTRADICTED: misstates something present in the passage / continuations.
- FABRICATED: invents specific entities/facts/events grounded in NONE of the passage, the corpus continuation, or the model's own generations.
- META: only describes genre/tone/style/structure — nothing concrete to check.

Think briefly, then answer with one word in <answer> tags, e.g. <answer>SUPPORTED</answer>."""


def ask(prompt):
    k = hashlib.sha1((PROMPT_VERSION + "|" + JUDGE + "|" + prompt).encode()).hexdigest()
    if k in cache and cache[k] is not None:
        return k, cache[k]
    body = {"model": JUDGE, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            m = ANS.search(r["choices"][0]["message"]["content"])
            if m:
                return k, m.group(1).upper()
        except Exception:
            continue
    return k, None


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def ols(sup, grp):
    rows = [(p, m, 0) for p, m in sup] + [(it["pos"], it["mg"], 1) for it in grp]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((res @ res) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), float(2 * stats.t.sf(abs(b[-1] / se), dof))


def main():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
    gen = {g["ci"]: g for g in json.load(open(HERE / "results" / "model_continuations.json"))}

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

    jobs, meta = [], []
    for arm in ("mat", "std"):
        for it in items[arm]:
            if it["v"] in H:
                g = gen[it["ci"]]
                mg_txt = "greedy: " + g["greedy"] + "\n" + "\n".join(f"sample {i+1}: {s}" for i, s in enumerate(g["samples"]))
                jobs.append(PROMPT.format(passage=man[it["ci"]]["prefix_text"],
                                          continuation=man[it["ci"]]["answer_text"],
                                          modelgen=mg_txt, note=it["text"]))
                meta.append(it["key"])
    print(f"[judge] re-judging {len(jobs)} hallucinated items WITH model's own generation…", flush=True)
    newv = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "150"))) as ex:
        res = []
        for c0 in range(0, len(jobs), 500):
            res += list(ex.map(ask, jobs[c0:c0 + 500]))
            for kk, vv in res[c0:]:
                cache[kk] = vv
            json.dump(cache, open(CACHE, "w"))
            print(f"  {min(c0 + 500, len(jobs))}/{len(jobs)}", flush=True)
    for key, (_, v) in zip(meta, res):
        newv[key] = v
    json.dump(newv, open(HERE / "results" / "withgen_verdicts.json", "w"))
    print(f"[judge] done, {sum(v is None for v in newv.values())} unparsed", flush=True)

    print("\n" + "=" * 74)
    out = {}
    rng = random.Random(0)
    for arm, label in [("mat", "matryoshka (lines)"), ("std", "standard (sentences)")]:
        rows = items[arm]
        sup = [(it["pos"], it["mg"]) for it in rows if it["v"] == "SUPPORTED"]
        loose = [it for it in rows if it["v"] in H]
        flipped = [it for it in loose if newv.get(it["key"]) == "SUPPORTED"]
        still = [it for it in loose if newv.get(it["key"]) in H]
        print(f"\n{label}: loose-hallucinated n={len(loose)}")
        print(f"  flip to SUPPORTED given model's own output: {len(flipped)} ({len(flipped)/len(loose):.0%})")
        print(f"  still hallucinated (given model output): {len(still)} ({len(still)/len(loose):.0%})")
        b0, p0 = ols(sup, loose); b1, p1 = ols(sup, still)
        print(f"  within-pos vs SUPPORTED — loose {b0:+.4f} (p={p0:.2f})  |  residual real {b1:+.4f} (p={p1:.2f}, n={len(still)})")
        real = b1
        perm = []
        for _ in range(800):
            idx = list(range(len(loose))); rng.shuffle(idx)
            perm.append(ols(sup, [loose[i] for i in idx[:len(still)]])[0])
        perm = np.array(perm); pperm = float(np.mean(np.abs(perm) >= abs(real)))
        print(f"  permutation (residual vs random same-size subset of loose): p_perm={pperm:.3f}")
        out[arm] = dict(n_loose=len(loose), n_flipped=len(flipped), n_still=len(still),
                        coef_residual=[b1, p1], perm_p=pperm)
    json.dump(out, open(HERE / "results" / "withgen_stats.json", "w"), indent=1)
    print("\n[saved] results/withgen_verdicts.json, withgen_stats.json")


if __name__ == "__main__":
    main()
