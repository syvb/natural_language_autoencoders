"""Word-chunk control for the ratio test: re-judge kitft with 10 equal-word chunks.

kitft emits ~3 long lines vs v3's ~10 short ones, so line-level order readout is
coarser for kitft (10% same-line ties vs 3%). This control removes the granularity
confound: each kitft explanation's full text is split into 10 equal-ish word chunks
(same convention as caa_steering_v2/judge_and_plot_wordchunk.py) and every chunk is
judged with the SAME two-trait 3-way rubric used on lines, giving kitft the same
order resolution as v3.

Reads results/ratio_raw_kitft.json, writes results/ratio_judged_kitft_chunks.json.
Run locally: python3 judge_ratio_chunks.py   (needs ~/.openrouter_key)
"""
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
N_CHUNKS = 10
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
CACHE = f"{HERE}/results/.ratio_linejudge_cache.json"
PROMPT_VERSION = "ratio2trait-v1"  # same rubric + cache namespace as judge_ratio.py

# PROMPT must stay byte-identical to judge_ratio.PROMPT (same labels, same cache
# semantics); judge_ratio.py can't be imported — its judging loop runs at module level.
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


def chunks_of(items, n=N_CHUNKS):
    words = " ".join(items).split()
    if not words:
        return []
    bounds = [round(i * len(words) / n) for i in range(n + 1)]
    return [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:]) if b > a]


rows = json.load(open(f"{HERE}/results/ratio_raw_kitft.json"))
todo = [chunks_of(x["items"]) for x in rows]
print(f"{len(rows)} rows, chunk counts: min {min(map(len,todo))}, max {max(map(len,todo))}")
with ThreadPoolExecutor(max_workers=32) as ex:
    results = list(ex.map(ask, todo))
fails = 0
out = []
for x, ch, (k, labels) in zip(rows, todo, results):
    cache[k] = labels
    if labels is None:
        fails += 1
        continue
    out.append({"R": x["R"], "lam": x["lam"], "rep": x["rep"], "base_idx": x["base_idx"],
                "n_items": len(ch), "yellow": labels["yellow"], "syco": labels["syco"]})
json.dump(cache, open(CACHE, "w"))
json.dump(out, open(f"{HERE}/results/ratio_judged_kitft_chunks.json", "w"))
print(f"kitft-chunks: {len(out)} rows judged ({fails} failures)")
print("RATIO_CHUNK_JUDGE_DONE")
