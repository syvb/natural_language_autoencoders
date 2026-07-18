"""Score truncated-steering fidelity from st_surgery.json (mat vs std).

Fidelity(k) = how well steering with the FIRST k explanation units reproduces
steering with ALL units. Two measures, seed-matched:

  textual  : token-level agreement between the k-steer and all-steer
             continuations (identical vectors -> identical T=1 text, so closer
             re-encodings share a longer prefix / more tokens). Judge-free.
  semantic : nex-n2-mini rates 0-3 whether the two continuations describe the
             same continuation direction.

Also reports the no-judge activation proxies already in the file: cos(v̂_k,v̂_all)
and FVE(v̂_k, v). Run: python judge_steer.py st_surgery.json
"""
import json, os, sys, re, time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

OR_KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
MODEL = "nex-agi/nex-n2-mini"


def toks(s):
    return re.findall(r"\S+", s)


def textual_agree(a, b):
    """Fraction of matching tokens over the shorter length, position-aligned
    (seed-matched T=1: same vector => same tokens; closer => longer match).
    Both empty => the two steers produced the same (null) behaviour => 1."""
    ta, tb = toks(a), toks(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    n = min(len(ta), len(tb))
    match = sum(1 for i in range(n) if ta[i] == tb[i])
    return match / max(len(ta), len(tb))


def judge(a, b, tries=4):
    prompt = (
        "Two continuations were generated from the SAME text prefix under the same "
        "random seed; only an internal steering vector differed. Rate how similar "
        "their CONTENT and DIRECTION are, ignoring length.\n"
        "3 = essentially the same continuation (same events/topic/tone)\n"
        "2 = clearly related, same general direction, some divergence\n"
        "1 = loosely related, mostly diverges\n"
        "0 = unrelated\n\n"
        f"Continuation A:\n{a[:600]}\n\nContinuation B:\n{b[:600]}\n\n"
        "Answer with ONLY the single digit 0, 1, 2, or 3.")
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 8}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=60))
            txt = r["choices"][0]["message"]["content"]
            m = re.search(r"[0-3]", txt)
            if m:
                return int(m.group())
        except Exception as e:
            time.sleep(2 * (t + 1))
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "st_surgery.json"
    data = json.load(open(path))
    K = [k for k in data["K_LIST"] if k != 0]  # truncation levels (exclude all)
    rows = data["results"]
    # collect all judge pairs first, then run them through a thread pool
    pairs = set()
    for row in rows:
        for m in ("mat", "std"):
            arms = row["models"][m]["arms"]
            for k in K:
                for gk, gall in zip(arms[str(k)]["gen"], arms["0"]["gen"]):
                    if gk.strip() and gall.strip() and gk != gall:
                        pairs.add((gk, gall))
    pairs = sorted(pairs)
    print(f"[judge] {len(pairs)} unique pairs to judge", flush=True)
    cache = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for pair, verdict in zip(pairs, ex.map(lambda p: judge(*p), pairs)):
            cache[pair] = verdict
    print(f"[judge] done; {sum(1 for v in cache.values() if v is None)} failures", flush=True)

    # aggregate
    agg = {m: {k: {"txt": [], "sem": [], "cos": [], "fve": []} for k in K} for m in ("mat", "std")}
    full_fve = {m: [] for m in ("mat", "std")}
    for row in rows:
        for m in ("mat", "std"):
            md = row["models"][m]
            arms = md["arms"]
            full_fve[m].append(arms["0"]["fve"])
            for k in K:
                ak, aall = arms[str(k)], arms["0"]
                agg[m][k]["cos"].append(ak["cos_all"])
                agg[m][k]["fve"].append(ak["fve"])
                for gk, gall in zip(ak["gen"], aall["gen"]):  # seed-matched
                    agg[m][k]["txt"].append(textual_agree(gk, gall))
                    if not gk.strip() and not gall.strip():
                        agg[m][k]["sem"].append(1.0)  # same null behaviour
                        continue
                    if not gk.strip() or not gall.strip():
                        agg[m][k]["sem"].append(0.0)
                        continue
                    if gk == gall:
                        agg[m][k]["sem"].append(1.0)
                        continue
                    v = cache.get((gk, gall))
                    if v is not None:
                        agg[m][k]["sem"].append(v / 3.0)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    print(f"{'model':>4} {'k':>3} | {'textual':>8} {'semantic':>8} | {'cos(v̂k,v̂all)':>13} {'FVE(v̂k,v)':>10}  n")
    out = {"model": {}}
    for m in ("mat", "std"):
        out["model"][m] = {"full_fve": round(mean(full_fve[m]), 4), "k": {}}
        print(f"  [{m}] full-explanation FVE = {mean(full_fve[m]):.3f}")
        for k in K:
            a = agg[m][k]
            r = dict(textual=round(mean(a["txt"]), 4), semantic=round(mean(a["sem"]), 4),
                     cos_all=round(mean(a["cos"]), 4), fve=round(mean(a["fve"]), 4),
                     n=len(a["txt"]))
            out["model"][m]["k"][k] = r
            print(f"{m:>4} {k:>3} | {r['textual']:>8.3f} {r['semantic']:>8.3f} | "
                  f"{r['cos_all']:>13.3f} {r['fve']:>10.3f}  {r['n']}")
    json.dump(out, open("steer_fidelity.json", "w"), indent=1)
    print("\n[saved] steer_fidelity.json")

    # headline: fidelity at k=1 and k=2
    for k in (1, 2):
        if k in K:
            print(f"\n  k={k}: matryoshka textual {out['model']['mat']['k'][k]['textual']:.2f} "
                  f"semantic {out['model']['mat']['k'][k]['semantic']:.2f}  vs  "
                  f"standard textual {out['model']['std']['k'][k]['textual']:.2f} "
                  f"semantic {out['model']['std']['k'][k]['semantic']:.2f}")


if __name__ == "__main__":
    main()
