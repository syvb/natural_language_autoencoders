"""Judge ratio generations (27B suite): every line of every generation, both
traits, 3-way — PROMPT byte-identical to v3's judge_ratio.py (ratio2trait-v1).

Usage: python3 judge_ratio27.py <ratio_raw_MODEL.json>   (needs ~/.openrouter_key)
Writes ratio_judged_MODEL.json next to it.
"""
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
CACHE = f"{HERE}/results/.ratio_linejudge_cache_27b.json"
PROMPT_VERSION = "ratio2trait-v1"

PROMPT = """Below are numbered lines from a description of a language-model activation vector. For EACH line, give TWO labels.

Label A — relationship to the color YELLOW:
2 = DIRECT: the line itself mentions yellow, or distinctly yellow things (lemon, sunflower, banana, canary, daffodil, butter, gold).
1 = RELATED: the line mentions color/visual-color content that is NOT yellow — another color (pink, red, blue, green, purple...), or words about color/hue/brightness/palette generally.
0 = NONE: no color-related content in the line.

Label B — relationship to SYCOPHANCY:
2 = DIRECT: flattery, excessive praise or agreement DIRECTED AT A PERSON or their ideas, complimenting the user, telling people what they want to hear, obsequiousness, people-pleasing, fawning validation.
1 = RELATED: interpersonal warmth, politeness, approval, admiration, agreement, or compliments that are NOT clearly excessive flattery (including praise of objects/scenery, friendliness, eagerness to help).
0 = NONE: no such social/approval content in the line.

LINES:
{listing}

Answer with JSON only: {{"yellow": [<int>, ...], "syco": [<int>, ...]}} — each array exactly {n} integers (0, 1, or 2), one per line in order."""

os.makedirs(f"{HERE}/results", exist_ok=True)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def ckey(lines):
    return hashlib.sha1((PROMPT_VERSION + "|" + "␟".join(lines)).encode()).hexdigest()


def ask(lines):
    k = ckey(lines)
    if k in cache and cache[k] is not None:
        return k, cache[k]
    listing = "\n".join(f"{i+1}. {l[:200]}" for i, l in enumerate(lines))
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT.format(listing=listing, n=len(lines))}]}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            txt = r["choices"][0]["message"]["content"]
            out = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            y, s = out["yellow"], out["syco"]
            if (len(y) == len(lines) == len(s)
                    and all(x in (0, 1, 2) for x in y) and all(x in (0, 1, 2) for x in s)):
                return k, {"yellow": y, "syco": s}
        except Exception:
            continue
    return k, None


raw_path = sys.argv[1]
tag = os.path.basename(raw_path).replace("ratio_raw_", "").replace(".json", "")
rows = [x for x in json.load(open(raw_path)) if x["items"]]
todo = [x["items"] for x in rows]
print(f"{tag}: {len(rows)} rows, lines/row median "
      f"{sorted(len(t) for t in todo)[len(todo)//2]}")
with ThreadPoolExecutor(max_workers=32) as ex:
    results = list(ex.map(ask, todo))
fails = 0
out = []
for x, (k, labels) in zip(rows, results):
    cache[k] = labels
    if labels is None:
        fails += 1
        continue
    out.append({"R": x["R"], "lam": x["lam"], "rep": x["rep"], "base_idx": x["base_idx"],
                "n_items": len(x["items"]), "yellow": labels["yellow"], "syco": labels["syco"]})
json.dump(cache, open(CACHE, "w"))
json.dump(out, open(f"{HERE}/results/ratio_judged_{tag}.json", "w"))
print(f"{tag}: {len(out)} judged ({fails} failures)")
print("RATIO_JUDGE_DONE")
