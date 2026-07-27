"""Token-decile chunking + faithfulness judging.

Removes the lines-vs-sentences granularity confound: split each explanation into
TEN equal token chunks (deciles) with the REAL Qwen tokenizer (not a word proxy),
judge each chunk for faithfulness, and report the hallucination rate by decile —
a clean, uniform position axis comparable across both models.

    JUDGE_WORKERS=200 python chunk_judge.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import analyze

HERE, SE = analyze.HERE, analyze.SE
KEY, URL, JUDGE = analyze.KEY, analyze.URL, analyze.JUDGE
H = {"CONTRADICTED", "FABRICATED"}
NCHUNK = 10
PROMPT_VERSION = "chunk-faith-v1"
CACHE = HERE / "results" / f".chunk_cache_{re.sub(r'[^a-z0-9]+','-',JUDGE.lower())}.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
ANS = re.compile(r"<answer>\s*(SUPPORTED|CONTRADICTED|FABRICATED|META)\s*</answer>", re.I)
BLUE, RED = "#2a78d6", "#e34948"

PROMPT = """An interpretability tool read a language model's internal state while it processed a passage and produced a description. You are shown a FRAGMENT of that description — a contiguous 10% slice of it by tokens, so it may begin or end mid-sentence. Judge only what THIS fragment asserts.

Passage (what the model had read):
<passage>
{passage}
</passage>
Text that actually came next:
<continuation>
{continuation}
</continuation>
Fragment of the description:
<fragment>
{fragment}
</fragment>

Classify the fragment:
- SUPPORTED: its concrete claims are consistent with the passage or the continuation.
- CONTRADICTED: it misstates a fact/entity/number/event from the passage or continuation.
- FABRICATED: it asserts specific entities/facts/events grounded in neither the passage nor the continuation.
- META: it contains no concrete checkable claim (pure genre/tone/structure, or a content-free fragment).

Answer with one word in <answer> tags, e.g. <answer>SUPPORTED</answer>."""


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


def get_tok(repo, sub):
    root = snapshot_download(repo, allow_patterns=[f"{sub}/*"],
                             token=open(os.path.expanduser("~/.hf_token")).read().strip())
    return AutoTokenizer.from_pretrained(f"{root}/{sub}")


def chunk10(text, tok):
    ids = tok.encode(text, add_special_tokens=False)
    n = len(ids)
    if n < NCHUNK:
        return None
    bnd = [round(i * n / NCHUNK) for i in range(NCHUNK + 1)]
    return [tok.decode(ids[bnd[i]:bnd[i + 1]]) for i in range(NCHUNK)]


def main():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    ARMS = [("mat", "ceselder/nla-qwen36-27b-matryoshka", "warmstart_av_lora",
             SE / "results" / "explanations_mat.json", BLUE),
            ("std", "ceselder/qwen3.6-27b-nla-L42", "av_sft_lora",
             SE / "results" / "explanations_std.json", RED)]

    # build all chunk jobs
    jobs, meta = [], []          # meta: (arm, ci, ri, decile)
    store = {}                   # (arm,ci,ri,decile) -> fragment text
    for arm, repo, sub, expl_path, _ in ARMS:
        tok = get_tok(repo, sub)
        expl = json.load(open(expl_path))["entries"]
        nskip = 0
        for e in expl:
            for ri, lines in enumerate(e["explanations"]):
                if not lines:
                    continue
                text = "\n".join(lines)
                chunks = chunk10(text, tok)
                if chunks is None:
                    nskip += 1
                    continue
                for d, frag in enumerate(chunks):
                    key = (arm, e["ci"], ri, d)
                    store[key] = frag
                    jobs.append(PROMPT.format(passage=man[e["ci"]]["prefix_text"],
                                              continuation=man[e["ci"]]["answer_text"], fragment=frag))
                    meta.append(key)
        print(f"[{arm}] {len([k for k in store if k[0]==arm])} chunks ({nskip} rollouts too short to chunk)", flush=True)

    print(f"[judge] {len(jobs)} chunk faithfulness calls…", flush=True)
    verd = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "200"))) as ex:
        res = []
        for c0 in range(0, len(jobs), 500):
            res += list(ex.map(ask, jobs[c0:c0 + 500]))
            for kk, vv in res[c0:]:
                cache[kk] = vv
            json.dump(cache, open(CACHE, "w"))
            print(f"  {min(c0 + 500, len(jobs))}/{len(jobs)}", flush=True)
    for key, (_, v) in zip(meta, res):
        verd[key] = v
    json.dump({"|".join(map(str, k)): v for k, v in verd.items()},
              open(HERE / "results" / "chunk_verdicts.json", "w"))
    print(f"[judge] done, {sum(v is None for v in verd.values())} unparsed", flush=True)

    # per-decile hallucination rate with cluster-bootstrap (over contexts) 95% CI
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for arm, repo, sub, expl_path, col in ARMS:
        # rows per context: list of (decile, is_halluc)
        by_ci = {}
        for (a, ci, ri, d), v in verd.items():
            if a != arm or v is None:
                continue
            by_ci.setdefault(ci, []).append((d, v in H))
        cis = list(by_ci)

        def rates(sample_cis):
            hit = np.zeros(NCHUNK); tot = np.zeros(NCHUNK)
            for c in sample_cis:
                for d, h in by_ci[c]:
                    tot[d] += 1; hit[d] += h
            return hit / np.maximum(tot, 1)

        point = rates(cis)
        boot = np.array([rates(rng.choices(cis, k=len(cis))) for _ in range(1500)])
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        x = np.arange(1, NCHUNK + 1)
        ax.errorbar(x, point, yerr=[point - lo, hi - point], fmt="o-", color=col, ms=6, lw=2,
                    capsize=3, label=f"{arm}")
        print(f"[{arm}] per-decile halluc rate: " + " ".join(f"{r:.2f}" for r in point))
    ax.set_xticks(range(1, NCHUNK + 1))
    ax.set_xlabel("explanation token decile (1 = first 10% of tokens → 10 = last 10%)", fontsize=11)
    ax.set_ylabel("hallucination rate of the chunk", fontsize=11)
    ax.set_title("Hallucination rate by token decile (real-tokenizer 10% chunks)\n"
                 "each explanation split into 10 equal token chunks, each judged", fontsize=12)
    ax.legend(fontsize=10, title="model")
    ax.grid(color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, -0.02,
             "Chunks are contiguous 10%-by-token slices (real Qwen tokenizer), so a chunk may start/end mid-sentence. "
             "Judged CON/FAB vs corpus continuation.\nError bars: cluster bootstrap over the 250 contexts (95%).",
             fontsize=7.7, color="#777", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "fig_chunk_halluc_rate.png", dpi=150, bbox_inches="tight")
    print("[saved] results/fig_chunk_halluc_rate.png")


if __name__ == "__main__":
    main()
