"""Fine-grained line-level judging for the YELLOW steering sweep.

Each line of each explanation is labeled:
  2 = directly mentions yellow / distinctly yellow things
  1 = related-but-not-yellow (another color or color-adjacent visual content)
  0 = no color-related content
Judged on the shared 41-of-82 r-subset for both models. Cached.
Writes yellow_linejudge_{v3,kitft}.json.
"""
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
CACHE = f"{HERE}/.yellow_linejudge_cache.json"
MAXL = 15

PROMPT = """Below are numbered lines from a description of a language-model activation vector. For EACH line, classify its relationship to the color YELLOW:

2 = DIRECT: the line itself mentions yellow, or distinctly yellow things (lemon, sunflower, banana, canary, daffodil, butter, gold).
1 = RELATED: the line mentions color/visual-color content that is NOT yellow — another color (pink, red, blue, green, purple...), or words about color/hue/brightness/palette generally.
0 = NONE: no color-related content in the line.

LINES:
{listing}

Answer with JSON only: {{"labels": [<int>, ...]}} with exactly {n} integers (0, 1, or 2), one per line in order."""

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def ckey(lines):
    return hashlib.sha1(("yellow3way|" + "␟".join(lines)).encode()).hexdigest()


def ask(lines):
    k = ckey(lines)
    if k in cache:
        return k, cache[k]
    listing = "\n".join(f"{i+1}. {l[:200]}" for i, l in enumerate(lines))
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT.format(listing=listing, n=len(lines))}]}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            txt = r["choices"][0]["message"]["content"]
            out = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])["labels"]
            if len(out) == len(lines) and all(x in (0, 1, 2) for x in out):
                return k, out
        except Exception:
            continue
    return k, None


for model, fn in (("v3", "frontload_v3_t1_raw.json"), ("kitft", "frontload_kitft_t1_raw.json")):
    rows = [x for x in json.load(open(f"{HERE}/{fn}")) if x["trait"] == "yellow"]
    all_r = sorted({x["r"] for x in rows})
    ri = np.linspace(0, len(all_r) - 1, 41).round().astype(int)
    rsel = {all_r[i] for i in ri}
    rows = [x for x in rows if x["r"] in rsel]
    todo = [x["items"][:MAXL] for x in rows]
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(ask, todo))
    fails = 0
    out = []
    for x, (k, labels) in zip(rows, results):
        cache[k] = labels
        if labels is None:
            fails += 1
            continue
        out.append({"r": x["r"], "base_idx": x["base_idx"], "n_items": x["n_items"],
                    "labels": labels})
    json.dump(cache, open(CACHE, "w"))
    json.dump(out, open(f"{HERE}/yellow_linejudge_{model}.json", "w"))
    print(f"{model}: {len(out)} rows judged ({fails} failures)", flush=True)
print("YELLOW_LINEJUDGE_DONE")
