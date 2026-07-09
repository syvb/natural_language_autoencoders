"""Word-chunk version of the first-mention-vs-steering-strength graph (v1 vs kitft).

Instead of splitting the AV explanation into newline items, split it into 10 EQUAL
word-chunks (deciles by word count) and ask the judge which chunk (1-10) first mentions
the trait (-1 none). This normalizes across models (v1 writes many short lines, kitft a
few long ones, but similar total words). Only 20 steering strengths are judged (subsampled
from the 82-point grid, shared across models) to bound cost; reuses existing AV explanations
(no GPU). Judged by Claude Haiku 4.5 via OpenRouter, cached.

One figure per trait, median AND mean of the 40 bases; non-appearance imputed as chunk 11.
"""
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
CACHE = os.path.join(R, ".judge_wordchunk_cache.json")
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
PROMPT_VERSION = "wordchunk-v1"
N_CHUNKS = 20 if False else 10   # 10 parts
N_POINTS = 20                    # 20 steering strengths
MISS = 11                        # non-appearance -> just past the 10 chunks
TRAITS = ["yellow", "sycophancy", "neuroticism"]
MODELS = [("v1 RLed NLA", "frontload_v2_raw_judged.json", "#1f77b4", "^-", 2.1),
          ("kitft (base verbalizer)", "frontload_kitft_raw_judged.json", "#888888", "s--", 1.9)]

DEFS = {
    "yellow": "the color YELLOW specifically, or distinctly yellow things (lemon, sunflower, banana, canary, daffodil, butter, gold). A DIFFERENT color (green, blue, orange, purple, red...) does NOT count.",
    "neuroticism": "an EMOTIONAL state of anxiety, worry, fear, nervousness, dread, depression, or emotional instability. Neutral difficulty or effort and ordinary description do NOT count.",
    "sycophancy": "flattery, excessive praise or agreement DIRECTED AT A PERSON or their ideas, complimenting the user, or obsequiousness. Describing scenery/objects as nice/beautiful, generic greetings or cheerfulness, and ordinary politeness/thanks do NOT count.",
}
PROMPT = """Below is a piece of text describing an activation vector, split into {n} numbered chunks (in order).

TRAIT ({trait}): {definition}

CHUNKS:
{listing}

Count a chunk ONLY if it ITSELF references the trait -- not if it merely expects/implies/precedes the trait, and not for pure register/format notes.

What is the number of the FIRST chunk that references the TRAIT? If none reference it, answer -1.
Answer with JSON only: {{"first_chunk": <integer>}}"""

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def chunks_of(text, n=N_CHUNKS):
    w = text.split()
    if not w:
        return [""] * n
    return [" ".join(c) for c in np.array_split(np.array(w, dtype=object), n)]


def ckey(trait, chunks):
    return hashlib.sha1((MODEL + "|" + PROMPT_VERSION + "|" + trait + "|" + "␟".join(chunks)).encode()).hexdigest()


def ask(args):
    trait, chunks = args
    k = ckey(trait, chunks)
    if k in cache:
        return k, cache[k]
    listing = "\n".join(f"{i+1}. {c}" for i, c in enumerate(chunks))
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT.format(n=N_CHUNKS, trait=trait, definition=DEFS[trait], listing=listing[:4000])}]}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=90))
            txt = r["choices"][0]["message"]["content"]
            i, j = txt.find("{"), txt.rfind("}")
            return k, int(json.loads(txt[i:j + 1])["first_chunk"])
        except Exception:
            continue
    return k, None


# ---- gather all rows needing judging (20 r-values, shared grid) ----
data = {}   # (mlabel, trait) -> list of (r, chunks)
todo = set()
for mlabel, fn, *_ in MODELS:
    rows = json.load(open(os.path.join(R, fn)))
    all_r = sorted({x["r"] for x in rows})
    ri = np.linspace(0, len(all_r) - 1, N_POINTS).round().astype(int)
    rsel = sorted({all_r[i] for i in ri})
    for trait in TRAITS:
        lst = []
        for x in rows:
            if x["trait"] != trait or x["r"] not in rsel:
                continue
            ch = chunks_of(x.get("explanation") or "\n".join(x["items"]))
            lst.append((x["r"], ch))
            if ckey(trait, ch) not in cache:
                todo.add((trait, tuple(ch)))
        data[(mlabel, trait)] = lst

print(f"judging {len(todo)} uncached word-chunk lists", flush=True)
if todo:
    with ThreadPoolExecutor(max_workers=12) as ex:
        for k, v in ex.map(ask, [(t, list(c)) for t, c in todo]):
            cache[k] = v
    json.dump(cache, open(CACHE, "w"))
print("judging done", flush=True)

# ---- plot: one figure per trait, v1 vs kitft, median + mean ----
for trait in TRAITS:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for mlabel, fn, color, style, lw in MODELS:
        byr = defaultdict(list)
        for r, ch in data[(mlabel, trait)]:
            fc = cache.get(ckey(trait, ch))
            byr[r].append(fc if (fc and fc >= 1) else MISS)
        rs = sorted(byr)
        med = [float(np.median(byr[r])) for r in rs]
        mean = [float(np.mean(byr[r])) for r in rs]
        ax.plot(rs, mean, style, color=color, lw=lw, ms=6, alpha=0.9, label=f"{mlabel} (mean)")
        ax.plot(rs, med, style, color=color, lw=lw * 0.6, ms=4, alpha=0.35, label=f"{mlabel} (median)")
    ax.set_xscale("log")
    ax.set_xlabel("steering strength  r  (log)")
    ax.set_ylabel(f"first word-chunk (of 10) mentioning {trait.upper()}   (none = 11)")
    ax.set_ylim(11.4, 0.6); ax.set_yticks(range(1, 12)); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Where steered {trait.upper()} first appears vs steering strength\n"
                 f"(AV output split into 10 equal word-chunks; {N_POINTS} strengths, 40 bases)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    out = os.path.join(R, f"fig_wordchunk_v1kitft_{trait}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))
print("WORDCHUNK_DONE")
