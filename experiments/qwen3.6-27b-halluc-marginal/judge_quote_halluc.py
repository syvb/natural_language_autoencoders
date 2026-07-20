"""Does the within-position marginal-FVE signal appear once we exclude
'unfaithful quote' hallucinations?

The faithfulness judge lumps two very different failures under
CONTRADICTED/FABRICATED: (a) an invented/misattributed QUOTATION (a quoted
string presented as being in the passage but that isn't), vs (b) a SUBSTANTIVE
fabrication/contradiction of facts in the note's own words. A fabricated quote
is specific, verbatim-looking text the critic reconstructs well regardless of
truth; a substantive prose fabrication may reconstruct poorly. A regex proxy
(quote-mark present) already splits the marginal sharply — this classifies the
REASON with the LLM to confirm.

For every hallucinated item (CONTRADICTED/FABRICATED), classify QUOTE vs
SUBSTANTIVE, then recompute the within-position OLS (marginal ~ C(pos)+is_h)
for SUBSTANTIVE-only hallucinations vs SUPPORTED.

    JUDGE_WORKERS=200 python judge_quote_halluc.py
"""
import hashlib
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

import analyze  # KEY, URL, JUDGE, SE, HERE

HERE = analyze.HERE
SE = analyze.SE
KEY, URL, JUDGE = analyze.KEY, analyze.URL, analyze.JUDGE
H = {"CONTRADICTED", "FABRICATED"}
PROMPT_VERSION = "quote-reason-v1"
_slug = re.sub(r"[^a-z0-9]+", "-", JUDGE.lower())
CACHE_PATH = HERE / "results" / f".quote_cache_{_slug}.json"
cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
ANS = re.compile(r"<answer>\s*(QUOTE|SUBSTANTIVE)\s*</answer>", re.I)

PROMPT = """An interpretability note about a text passage was already judged UNFAITHFUL (it misstates or invents something not supported by the passage). Your job is to classify WHY it is unfaithful.

Passage (the model had read this much):
<passage>
{passage}
</passage>

The text that actually came next:
<continuation>
{continuation}
</continuation>

The unfaithful note:
<note>
{note}
</note>

Classify:
- QUOTE: the unfaithfulness is essentially an invented or misattributed QUOTATION — a quoted string the note presents as appearing in the passage but that is not there verbatim. If you removed the quoted phrase, the note's own-words description would be roughly consistent with the passage.
- SUBSTANTIVE: the note invents or contradicts facts, entities, events, numbers, or relationships in its OWN words. (Choose this even if a quote is also present, as long as there is non-quote fabrication/contradiction.)

Answer with exactly one word in <answer> tags, e.g. <answer>QUOTE</answer>."""


def ask(prompt):
    k = hashlib.sha1((PROMPT_VERSION + "|" + JUDGE + "|" + prompt).encode()).hexdigest()
    if k in cache and cache[k] is not None:
        return k, cache[k]
    body = {"model": JUDGE, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            req = urllib.request.Request(
                URL, json.dumps(body).encode(),
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


def ols(sub):
    """coef,p,n for is_target in marginal ~ C(pos)+is_target; sub rows = (k,mg,is_target)."""
    ks = np.array([r[0] for r in sub]); y = np.array([r[1] for r in sub])
    h = np.array([r[2] for r in sub], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [h])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; dof = len(y) - X.shape[1]
    cov = (r @ r) / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[-1, -1]); t = b[-1] / se
    return float(b[-1]), float(2 * stats.t.sf(abs(t), dof)), len(y)


def main():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))

    # gather all items with (arm,ci,ri,k,pos,marginal,verdict,text)
    items = {}
    for arm, fn in [("mat", "subset_scores_mat.json"), ("std", "subset_scores_std.json")]:
        items[arm] = []
        for e in json.load(open(HERE / "results" / fn))["entries"]:
            for ri, rec in enumerate(e["rollouts"]):
                if not rec:
                    continue
                m = marg(rec)
                for k in range(len(m)):
                    v = faith.get(f"{arm}|{e['ci']}|{ri}|{k}")
                    if v is not None:
                        items[arm].append(dict(ci=e["ci"], ri=ri, k=k, pos=k,
                                               mg=m[k], v=v, text=rec["units"][k]))

    # classify every hallucinated item
    jobs, meta = [], []
    for arm in ("mat", "std"):
        for it in items[arm]:
            if it["v"] in H:
                key = f"{arm}|{it['ci']}|{it['ri']}|{it['k']}"
                jobs.append(PROMPT.format(passage=man[it["ci"]]["prefix_text"],
                                          continuation=man[it["ci"]]["answer_text"],
                                          note=it["text"]))
                meta.append(key)
    print(f"[judge] classifying {len(jobs)} hallucinated items QUOTE vs SUBSTANTIVE…", flush=True)
    reason = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "200"))) as ex:
        res = []
        for c0 in range(0, len(jobs), 500):
            res += list(ex.map(ask, jobs[c0:c0 + 500]))
            for kk, vv in res[c0:]:
                cache[kk] = vv
            json.dump(cache, open(CACHE_PATH, "w"))
            print(f"  {min(c0 + 500, len(jobs))}/{len(jobs)}", flush=True)
    for key, (_, v) in zip(meta, res):
        reason[key] = v
    fails = sum(1 for v in reason.values() if v is None)
    json.dump(reason, open(HERE / "results" / "quote_reason.json", "w"))
    print(f"[judge] done, {fails} unparsed", flush=True)

    # analysis: within-position OLS, SUBSTANTIVE-halluc vs SUPPORTED (and QUOTE vs SUP)
    out = {}
    print("\n" + "=" * 74)
    for arm, label in [("mat", "matryoshka (lines)"), ("std", "standard (sentences)")]:
        rws = items[arm]
        sup = [(it["pos"], it["mg"], False) for it in rws if it["v"] == "SUPPORTED"]
        subst, quote = [], []
        for it in rws:
            if it["v"] in H:
                r = reason.get(f"{arm}|{it['ci']}|{it['ri']}|{it['k']}")
                (subst if r == "SUBSTANTIVE" else quote if r == "QUOTE" else []).append(
                    (it["pos"], it["mg"], True))
        c_sub, p_sub, n_sub = ols(subst + sup)
        c_q, p_q, n_q = ols(quote + sup)
        allh = [(it["pos"], it["mg"], True) for it in rws if it["v"] in H]
        c_all, p_all, _ = ols(allh + sup)
        m_sub = np.mean([r[1] for r in subst]); m_q = np.mean([r[1] for r in quote])
        m_sup = np.mean([r[1] for r in sup])
        print(f"\n{label}")
        print(f"  hallucinated split: SUBSTANTIVE {len(subst)} | QUOTE {len(quote)} | "
              f"(SUPPORTED {len(sup)})")
        print(f"  mean marginal: SUBSTANTIVE {m_sub:+.4f} | QUOTE {m_q:+.4f} | SUPPORTED {m_sup:+.4f}")
        print(f"  within-position OLS vs SUPPORTED:")
        print(f"    ALL hallucinated   coef {c_all:+.4f} p={p_all:.3f}")
        print(f"    SUBSTANTIVE only   coef {c_sub:+.4f} p={p_sub:.2e}")
        print(f"    QUOTE only         coef {c_q:+.4f} p={p_q:.3f}")
        out[arm] = dict(n_substantive=len(subst), n_quote=len(quote), n_supported=len(sup),
                        mean_substantive=float(m_sub), mean_quote=float(m_q), mean_supported=float(m_sup),
                        ols_all=[c_all, p_all], ols_substantive=[c_sub, p_sub], ols_quote=[c_q, p_q])
    json.dump(out, open(HERE / "results" / "quote_halluc_stats.json", "w"), indent=1)
    print("\n[saved] results/quote_reason.json, results/quote_halluc_stats.json")


if __name__ == "__main__":
    main()
