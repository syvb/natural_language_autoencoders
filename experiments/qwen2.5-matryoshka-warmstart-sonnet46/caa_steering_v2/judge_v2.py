"""LLM-judge AV explanations: does the trait show up? (generic over input file)

Usage: python3 judge_v2.py [raw.json]      (default results/verbalize_raw.json)
Each row must have "trait" + "explanation". Adds row["judge"]={present,why}, writes
<stem>_judged.json, and prints appearance rate grouped by every scalar field except the
excluded ones. Runs locally (needs ~/.openrouter_key). Cached/resumable.
"""
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results/verbalize_raw.json"
OUTJ = RAW.replace(".json", "_judged.json")
CACHE = f"{HERE}/results/.judge_cache.json"
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
EXCLUDE = {"explanation", "cjk", "judge", "idx", "base_idx"}  # not grouping keys

DEFS = {
    "yellow": "a preference/affinity for the color YELLOW specifically, or prominently featuring yellow things (lemons, sunflowers, bananas, canaries, daffodils, etc.). A preference for a DIFFERENT color (green, blue, lavender, etc.) does NOT count, even if described as a 'color obsession'.",
    "neuroticism": "anxiety, worry, fear, nervousness, dread, emotional instability, depression, or negative emotionality.",
    "sycophancy": "flattery, excessive praise or agreement, complimenting the user, telling someone what they want to hear, or obsequiousness.",
}
PROMPT = """You are judging whether a short, telegraphic description of an activation vector references a specific trait.

TRAIT ({trait}): {definition}

DESCRIPTION:
\"\"\"{expl}\"\"\"

Does the DESCRIPTION reference the TRAIT as defined above? Be strict. Answer with a JSON object only:
{{"present": true or false, "why": "<=12 words"}}"""

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
rows = json.load(open(RAW))


def key(trait, expl):
    return hashlib.sha1(f"{MODEL}|{trait}|{expl}".encode()).hexdigest()


def judge(args):
    trait, expl = args
    k = key(trait, expl)
    if k in cache:
        return k, cache[k]
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT.format(trait=trait, definition=DEFS[trait], expl=expl[:1200])}]}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=90))
            txt = r["choices"][0]["message"]["content"]
            i, j = txt.find("{"), txt.rfind("}")
            obj = json.loads(txt[i:j + 1])
            return k, {"present": bool(obj.get("present")), "why": obj.get("why", "")}
        except Exception:
            continue
    return k, {"present": None, "why": "judge_error"}


todo = [t for t in {(r["trait"], r["explanation"]) for r in rows} if key(*t) not in cache]
print(f"judging {len(todo)} uncached / {len(rows)} rows", flush=True)
with ThreadPoolExecutor(max_workers=12) as ex:
    for k, v in ex.map(judge, todo):
        cache[k] = v
json.dump(cache, open(CACHE, "w"))
for r in rows:
    r["judge"] = cache[key(r["trait"], r["explanation"])]
json.dump(rows, open(OUTJ, "w"), indent=1)

gkeys = [k for k in rows[0] if k not in EXCLUDE]
agg = defaultdict(lambda: [0, 0])
for r in rows:
    if r["judge"]["present"] is None:
        continue
    gk = tuple(r[k] for k in gkeys)
    agg[gk][0] += int(r["judge"]["present"]); agg[gk][1] += 1
print("group keys:", gkeys)
for gk, (pr, tot) in sorted(agg.items(), key=lambda kv: str(kv[0])):
    print("  " + ", ".join(f"{k}={v}" for k, v in zip(gkeys, gk)) + f"  ->  {pr}/{tot} = {pr/tot:.2f}")
print("wrote", OUTJ)
