"""Find the 1-based index of the first AV list item that mentions the trait.

Reads frontload_v2_raw.json (each row has an `items` list). Asks Claude Haiku 4.5 for the first
item index referencing the trait (-1 if none), caches, then aggregates first-index & normalized
rank vs steering strength r (over rows where the trait appears) and prints Spearman(r, first_index).

Run locally (needs ~/.openrouter_key). Usage: python3 judge_first_index.py
"""
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results/frontload_v2_raw.json"
OUTJ = RAW.replace(".json", "_judged.json")
CACHE = f"{HERE}/results/.judge_idx_cache.json"
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"

DEFS = {
    "yellow": "the color YELLOW specifically, or distinctly yellow things (lemon, sunflower, banana, canary, daffodil, butter, gold). A DIFFERENT color (green, blue, orange, purple, red...) does NOT count.",
    "neuroticism": "an EMOTIONAL state of anxiety, worry, fear, nervousness, dread, depression, or emotional instability. Neutral difficulty or effort ('hard to balance', 'I struggle with X') and ordinary description do NOT count.",
    "sycophancy": "flattery, excessive praise or agreement DIRECTED AT A PERSON or their ideas, complimenting the user, or obsequiousness. Describing scenery/objects as nice/beautiful, generic greetings or cheerfulness, and ordinary politeness/thanks do NOT count.",
}
PROMPT_VERSION = "v2"
PROMPT = """Below is a numbered list of short snippets describing an activation vector.

TRAIT ({trait}): {definition}

LIST:
{listing}

Count a snippet ONLY if it ITSELF references the trait -- not if it merely "expects"/"suggests"/"implies"/"precedes" the trait appearing next, and not if it is a pure register/format note (e.g. "English learner register", "expects closing phrase").

What is the number of the FIRST list item that references the TRAIT? If no item references it, answer -1.
Answer with JSON only: {{"first_index": <integer>}}"""

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
rows = json.load(open(RAW))


def key(trait, items):
    return hashlib.sha1(("{}|{}|{}|".format(MODEL, PROMPT_VERSION, trait) + "\n".join(items)).encode()).hexdigest()


def ask(args):
    trait, items = args
    k = key(trait, items)
    if k in cache:
        return k, cache[k]
    listing = "\n".join(f"{i+1}. {it}" for i, it in enumerate(items))
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT.format(trait=trait, definition=DEFS[trait], listing=listing[:4000])}]}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=90))
            txt = r["choices"][0]["message"]["content"]
            i, j = txt.find("{"), txt.rfind("}")
            return k, int(json.loads(txt[i:j + 1])["first_index"])
        except Exception:
            continue
    return k, None


todo = [(r["trait"], tuple(r["items"])) for r in rows if r["items"] and key(r["trait"], r["items"]) not in cache]
todo = list({(t, it) for t, it in todo})
print(f"judging {len(todo)} uncached lists", flush=True)
with ThreadPoolExecutor(max_workers=12) as ex:
    for k, v in ex.map(ask, [(t, list(it)) for t, it in todo]):
        cache[k] = v
json.dump(cache, open(CACHE, "w"))

for r in rows:
    r["first_index"] = cache.get(key(r["trait"], r["items"])) if r["items"] else None
json.dump(rows, open(OUTJ, "w"), indent=1)


def spearman(x, y):
    import numpy as np
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


import numpy as np
print("\n=== first-mention list index vs steering strength r ===")
agg = defaultdict(lambda: {"idx": [], "norm": [], "app": 0, "tot": 0})
allpts = defaultdict(lambda: ([], []))  # trait -> (r_list, idx_list) over appearing rows
for r in rows:
    fi = r.get("first_index")
    a = agg[(r["trait"], r["r"])]
    a["tot"] += 1
    if fi and fi >= 1:
        a["app"] += 1
        a["idx"].append(fi)
        a["norm"].append(fi / max(1, r["n_items"]))
        allpts[r["trait"]][0].append(r["r"]); allpts[r["trait"]][1].append(fi)
RS_ALL = sorted({r["r"] for r in rows})
for trait in ["sycophancy", "neuroticism", "yellow"]:
    print(f"\n{trait}:   (appear_rate | median_idx | mean_idx | mean_norm_rank)")
    for rr in RS_ALL:
        a = agg[(trait, rr)]
        if a["tot"]:
            mi = f"{int(np.median(a['idx']))}" if a["idx"] else "-"
            me = f"{np.mean(a['idx']):.1f}" if a["idx"] else "-"
            nr = f"{np.mean(a['norm']):.2f}" if a["norm"] else "-"
            print(f"  r={rr:<5} app={a['app']:>2}/{a['tot']}  med_idx={mi:>3}  mean_idx={me:>4}  norm={nr}")
    xr, xi = allpts[trait]
    print(f"  Spearman(r, first_index) over appearing rows = {spearman(xr, xi):+.3f}  (n={len(xr)})")
print("\nwrote", OUTJ)
