"""Which token categories have different marginal FVE, controlling for token index?

For each category: index-stratified excess = weighted mean over index bins of
(mean ΔFVE of category tokens in bin − mean ΔFVE of non-category tokens in bin),
weights = category token count per bin. Bootstrap CI over the 100 samples.
Also reports the same statistic additionally stratified by in-quote membership
(category effects can just proxy for quotes otherwise).
"""
import json
import re

import numpy as np

D = "/home/debian/nla-doll/experiments/qwen2.5-matryoshka-warmstart-sonnet46/v3_faithfulness_results"
BINS = [0, 5, 10, 20, 40, 80, 160, 10 ** 9]

STOP = set("""the a an of to in and or is are was were be been being as at by for from
with that this these those it its their his her which who whom what on but not no
if then than so such can could would should may might will shall do does did have
has had s t d ll re ve m""".split())
SCHEMA = set("""token tokens pattern domain register tone genre structure sentence
clause list item expect expects expected expectation signal signals signalling
continues continuation continuing likely next final format style narrative text
context phrase word words paragraph topic theme voice tense mid grammatical
syntax noun verb subject predict prediction incomplete ongoing""".split())
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


def categorize(tokens, in_quote):
    """Return dict cat -> bool array."""
    n = len(tokens)
    cats = {}
    strip = [t.strip() for t in tokens]
    low = [s.lower() for s in strip]
    cats["quote_mark"] = np.array(['"' in t for t in tokens])
    cats["newline"] = np.array(["\n" in t for t in tokens])
    line_first = np.zeros(n, bool)
    for i in range(1, n):
        if "\n" in tokens[i - 1]:
            line_first[i] = True
    line_first[0] = True
    cats["line_first"] = line_first
    cats["punct_only"] = np.array([bool(s) and not any(c.isalnum() for c in s) and '"' not in s and "\n" not in t
                                   for s, t in zip(strip, tokens)])
    cats["stopword"] = np.array([l in STOP for l in low])
    cats["numeric"] = np.array([any(c.isdigit() for c in s) for s in strip])
    cats["capitalized"] = np.array([bool(s) and s[0].isupper() for s in strip])
    cats["subword_cont"] = np.array([bool(t) and not t[0].isspace() and bool(strip[i]) and strip[i][0].isalnum() and i > 0
                                     and not ("\n" in tokens[i - 1])
                                     for i, t in enumerate(tokens)])
    cats["cjk"] = np.array([bool(CJK.search(t)) for t in tokens])
    cats["schema_word"] = np.array([l in SCHEMA for l in low])
    after_q = np.zeros(n, bool)
    for i in range(1, n):
        if '"' in tokens[i - 1]:
            after_q[i] = True
    cats["after_quote_tok"] = after_q
    line_last = np.zeros(n, bool)
    for i in range(n - 1):
        if "\n" in tokens[i + 1]:
            line_last[i] = True
    cats["line_last"] = line_last
    cats["in_quote"] = np.array(in_quote, bool)
    return cats


def load(path):
    rows = json.load(open(path))
    per_sample = []
    for r in rows:
        cats = categorize(r["tokens"], r["in_quote"])
        idx = np.arange(len(r["tokens"]))
        d = np.array(r["delta_fve"])
        per_sample.append((idx, d, cats, np.array(r["in_quote"], bool)))
    return per_sample


def excess(samples, cat, quote_stratum=None, sample_ids=None):
    """Index-stratified (and optionally quote-stratified) excess ΔFVE for cat."""
    ids = range(len(samples)) if sample_ids is None else sample_ids
    binof = lambda i: np.digitize(i, BINS) - 1
    # strata key: (index bin, quote flag if stratifying)
    cat_sum, cat_cnt, oth_sum, oth_cnt = {}, {}, {}, {}
    for si in ids:
        idx, d, cats, inq = samples[si]
        c = cats[cat]
        if quote_stratum is not None:
            keep = inq == quote_stratum
        else:
            keep = np.ones(len(idx), bool)
        for stratum in set(binof(idx[keep])):
            m = keep & (binof(idx) == stratum)
            cm, om = m & c, m & ~c
            cat_sum[stratum] = cat_sum.get(stratum, 0) + d[cm].sum()
            cat_cnt[stratum] = cat_cnt.get(stratum, 0) + cm.sum()
            oth_sum[stratum] = oth_sum.get(stratum, 0) + d[om].sum()
            oth_cnt[stratum] = oth_cnt.get(stratum, 0) + om.sum()
    num = den = 0.0
    tot_cat = 0
    for s in cat_cnt:
        if cat_cnt[s] < 5 or oth_cnt[s] < 5:
            continue
        w = cat_cnt[s]
        num += w * (cat_sum[s] / cat_cnt[s] - oth_sum[s] / oth_cnt[s])
        den += w
        tot_cat += cat_cnt[s]
    return (num / den if den else float("nan")), tot_cat


def analyze(name, path):
    samples = load(path)
    ncat = {c: sum(s[2][c].sum() for s in samples) for c in samples[0][2]}
    ntot = sum(len(s[0]) for s in samples)
    print(f"\n=== {name} ({ntot} tokens, 100 samples) ===")
    print(f"{'category':1s}".ljust(16) + f"{'%tok':>6s} {'excess ΔFVE':>12s} {'95% CI':>20s} {'|inQ':>9s} {'|outQ':>9s}")
    rng = np.random.default_rng(0)
    out = {}
    for cat in samples[0][2]:
        e, n = excess(samples, cat)
        boots = []
        for _ in range(300):
            ids = rng.integers(0, len(samples), len(samples))
            b, _ = excess(samples, cat, sample_ids=ids)
            if b == b:
                boots.append(b)
        if not boots:
            print(f"{cat:16s}{ncat[cat]/ntot:>6.1%} {'—':>12s}  [too few tokens]")
            continue
        lo, hi = np.percentile(boots, [2.5, 97.5])
        eq, _ = excess(samples, cat, quote_stratum=True)
        en, _ = excess(samples, cat, quote_stratum=False)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{cat:16s}{ncat[cat]/ntot:>6.1%} {e:>+12.5f}{sig} [{lo:>+8.5f},{hi:>+8.5f}] {eq:>+9.5f} {en:>+9.5f}")
        out[cat] = dict(frac=ncat[cat] / ntot, excess=e, lo=lo, hi=hi, in_q=eq, out_q=en)
    return out

res = {"v3rl": analyze("v3 RL", f"{D}/quote_marginal_v3rl.json"),
       "v3ws": analyze("v3 warm-start", f"{D}/quote_marginal_v3ws.json")}
json.dump(res, open(f"{D}/token_category_analysis.json", "w"), indent=1)
print("\nwrote token_category_analysis.json")
