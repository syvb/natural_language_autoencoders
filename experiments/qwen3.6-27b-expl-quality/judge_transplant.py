"""Format-transplant probe: break the style confound (LOCAL, OpenRouter).

The main eval's pairwise preferences are style-contaminated (deranged std
prose still scores 4.91 coherence). This probe measures the FORMAT main
effect directly, with a usefulness-ONLY prompt that deliberately does not
over-specify what usefulness means — only that it is about understanding the
model's internals, not readability.

  absolute (4 arms x 500 contexts): each explanation judged in its NATIVE
    format and TRANSPLANTED into the other one —
      mat/list (native)   mat/prose (lines joined into a paragraph)
      std/prose (native)  std/list (sentences, one per line)
    Content is byte-identical across the two renderings of an explanation;
    any score gap is pure format.
  paired, format-equalized (2 formats x 2 orders x 500): mat vs std with BOTH
    rendered in the SAME format — the style-neutralized head-to-head.

    python judge_transplant.py                # gemma + CoreWeave (pinned)

Same judge-swap/caching machinery as judge_quality.py (per-judge cache +
output, raw responses cached, atomic writes, 429 backoff).
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "expl-transplant-v1"

USE_RE = re.compile(r"<usefulness>\s*([1-5])\s*</usefulness>", re.I)
PAIR_RE = re.compile(r"<more_useful>\s*(A|B)\s*</more_useful>", re.I)
# quote-parity-aware sentence splitter (same family as the rev experiment)
SENT_RE = re.compile(r'[.!?]["”\')\]]*\s+(?=[A-Z"“(\d])')

INTRO = """An interpretability tool read a language model's internal activation, recorded at the final token of a passage the model was processing, and wrote an "explanation" of that activation.

Here is the passage (the model had read exactly this much; the activation is at its final token):

<passage>
{passage}
</passage>

The text that actually came next (the model had NOT seen this yet; the explanation may legitimately describe or predict it):

<continuation>
{continuation}
</continuation>"""

ABS_PROMPT = INTRO + """

The explanation:

<explanation>
{expl}
</explanation>

Rate the explanation's USEFULNESS from 1 (useless) to 5 (extremely useful).

Usefulness here means how much the explanation helps you understand the model's internals — what the model was processing at that moment and where it was headed. It is not about readability or writing style.

Think briefly, then answer in tags, e.g. <usefulness>3</usefulness>."""

PAIR_PROMPT = INTRO + """

Two different tools each explained the SAME activation:

<explanation_A>
{expl_a}
</explanation_A>

<explanation_B>
{expl_b}
</explanation_B>

Which explanation is more USEFUL? Usefulness here means how much an explanation helps you understand the model's internals — what the model was processing at that moment and where it was headed. It is not about readability or writing style.

You must choose one, even if it is close. Think briefly, then answer A or B in tags, e.g. <more_useful>A</more_useful>."""

NUDGE_ABS = "\n\nIMPORTANT: end your reply with the literal tag <usefulness>N</usefulness> where N is 1-5."
NUDGE_PAIR = "\n\nIMPORTANT: end your reply with the literal tag <more_useful>A</more_useful> (or B). You must pick one."


def sentences(text):
    parts, prev = [], 0
    for m in SENT_RE.finditer(text):
        cand = text[prev: m.end()].strip()
        if cand.count('"') % 2 == 0:            # don't split inside a quote
            parts.append(cand)
            prev = m.end()
    tail = text[prev:].strip()
    if tail:
        parts.append(tail)
    return parts


def as_list(arm, lines):
    """One unit per line. std's prose gets sentence-split first."""
    if arm == "std":
        units = [s for ln in lines for s in sentences(ln)]
    else:
        units = lines
    return "\n".join(units)


def as_prose(arm, lines):
    """One paragraph. mat's telegraphic lines get terminal periods."""
    if arm == "mat":
        units = [ln.strip() + ("" if ln.strip().endswith((".", "!", "?")) else ".")
                 for ln in lines]
        return " ".join(units)
    return " ".join(ln.strip() for ln in lines)


def parse_abs(txt):
    m = USE_RE.search(txt)
    return {"usefulness": int(m.group(1))} if m else None


def parse_pair(txt):
    m = PAIR_RE.search(txt)
    return {"more_useful": m.group(1).upper()} if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default=os.environ.get("JUDGE_MODEL",
                                                      "google/gemma-4-31b-it"))
    ap.add_argument("--provider", default=os.environ.get("JUDGE_PROVIDER",
                                                         "CoreWeave"))
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest_quality.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    judge = args.judge
    slug = re.sub(r"[^a-z0-9]+", "-", judge.lower()).strip("-")
    cache_path = HERE / "results" / f".transplant_cache_{slug}.json"
    out_path = HERE / "results" / f"transplant_{slug}.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    print(f"[judge] {judge} @ {args.provider} | cache {len(cache)}", flush=True)

    def flush_cache():
        tmp = cache_path.with_suffix(".tmp")
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, cache_path)

    def _call(prompt):
        body = {"model": judge, "temperature": 0, "stream": True,
                "messages": [{"role": "user", "content": prompt}]}
        if args.provider:
            body["provider"] = {"order": [args.provider],
                                "allow_fallbacks": False}
        req = urllib.request.Request(
            URL, json.dumps(body).encode(),
            {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
             "Accept": "text/event-stream"})
        chunks = []
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in j:
                    raise RuntimeError(str(j["error"]))
                d = (j.get("choices") or [{}])[0].get("delta") or {}
                if d.get("content"):
                    chunks.append(d["content"])
        return "".join(chunks)

    def ask(job):
        kind, prompt = job
        k = hashlib.sha1((PROMPT_VERSION + "|" + kind + "|" + prompt).encode()).hexdigest()
        parse = parse_abs if kind == "abs" else parse_pair
        nudge = NUDGE_ABS if kind == "abs" else NUDGE_PAIR
        if cache.get(k) is not None:
            p = parse(cache[k])
            if p is not None:
                return k, None, p
        txt = None
        for attempt in range(5):
            p = prompt + nudge if attempt > 2 else prompt
            try:
                txt = _call(p)
                if parse(txt) is not None:
                    break
            except urllib.error.HTTPError as e:
                time.sleep((20 if e.code == 429 else 5) * (attempt + 1)
                           + random.random() * 5)
            except Exception:
                time.sleep(5 * (attempt + 1))
        return k, txt, (parse(txt) if txt is not None else None)

    def run_jobs(jobs, tag):
        workers = int(os.environ.get("JUDGE_WORKERS", "200"))
        fails, res = 0, []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(ask, j) for j in jobs]
            for n, f in enumerate(futs, 1):
                kk, raw, v = f.result()
                if raw is not None:
                    cache[kk] = raw
                res.append(v)
                if v is None:
                    fails += 1
                if n % 50 == 0:
                    flush_cache()
                    print(f"[{tag}] {n}/{len(jobs)} ({fails} unparsed)", flush=True)
        flush_cache()
        print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
        return res

    man = {c["ci"]: c for c in json.load(open(args.manifest))["contexts"]}
    ex_ = {}
    for arm in ("mat", "std"):
        d = json.load(open(HERE / "results" / f"explanations_{arm}.json"))
        ex_[arm] = {e["ci"]: e["explanations"][0] for e in d["entries"]}
    cis = sorted(set(ex_["mat"]) & set(ex_["std"]) & set(man))
    if args.limit:
        cis = cis[: args.limit]
    dom = {ci: man[ci]["domain"] for ci in cis}
    print(f"[data] {len(cis)} contexts", flush=True)

    RENDER = {"list": as_list, "prose": as_prose}

    def abs_prompt(ci, arm, fmt):
        return ABS_PROMPT.format(passage=man[ci]["prefix_text"],
                                 continuation=man[ci]["continuation_text"],
                                 expl=RENDER[fmt](arm, ex_[arm][ci]))

    report = {"meta": {"judge": judge, "provider": args.provider,
                       "prompt_version": PROMPT_VERSION, "n_contexts": len(cis)}}

    # ── absolute: 2 arms x 2 formats ─────────────────────────────────────────
    jobs, idx = [], []
    for arm in ("mat", "std"):
        for fmt in ("list", "prose"):
            for ci in cis:
                jobs.append(("abs", abs_prompt(ci, arm, fmt)))
                idx.append((arm, fmt, ci))
    res = run_jobs(jobs, "absolute")
    report["absolute"] = [{"arm": a, "fmt": f, "ci": ci, **v}
                          for (a, f, ci), v in zip(idx, res) if v]

    # ── paired, format-equalized: both arms in the SAME format, both orders ──
    jobs, meta = [], []
    for fmt in ("list", "prose"):
        for ci in cis:
            for order, (fa, fb) in [("ms", ("mat", "std")), ("sm", ("std", "mat"))]:
                jobs.append(("pair", PAIR_PROMPT.format(
                    passage=man[ci]["prefix_text"],
                    continuation=man[ci]["continuation_text"],
                    expl_a=RENDER[fmt](fa, ex_[fa][ci]),
                    expl_b=RENDER[fmt](fb, ex_[fb][ci]))))
                meta.append((fmt, ci, order, fa, fb))
    res = run_jobs(jobs, "paired-eq")
    paired = []
    for (fmt, ci, order, fa, fb), v in zip(meta, res):
        if not v:
            continue
        ab = v["more_useful"]
        paired.append({"fmt": fmt, "ci": ci, "order": order,
                       "winner": "TIE" if ab == "TIE" else (fa if ab == "A" else fb)})
    report["paired"] = paired

    out_path.write_text(json.dumps(report))
    print(f"[write] {out_path}", flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    A = {(r["arm"], r["fmt"], r["ci"]): r["usefulness"] for r in report["absolute"]}

    def mean(vals):
        return sum(vals) / max(1, len(vals))

    for gname, gdom in [("all", None), ("pretrain", "pretrain"),
                        ("wildchat", "wildchat")]:
        g = [ci for ci in cis if gdom is None or dom[ci] == gdom]
        print(f"═══ {gname} ═══")
        for arm in ("mat", "std"):
            for fmt in ("list", "prose"):
                vals = [A[(arm, fmt, ci)] for ci in g if (arm, fmt, ci) in A]
                print(f"  use({arm}/{fmt}) = {mean(vals):.2f}  (n={len(vals)})")
        for arm in ("mat", "std"):
            d = [A[(arm, "prose", ci)] - A[(arm, "list", ci)] for ci in g
                 if (arm, "prose", ci) in A and (arm, "list", ci) in A]
            print(f"  FORMAT effect {arm} (prose−list): {mean(d):+.3f}")
        for fmt in ("list", "prose"):
            d = [A[("mat", fmt, ci)] - A[("std", fmt, ci)] for ci in g
                 if ("mat", fmt, ci) in A and ("std", fmt, ci) in A]
            print(f"  ARM effect in {fmt:5s} (mat−std): {mean(d):+.3f}")
        for fmt in ("list", "prose"):
            cnt = Counter(r["winner"] for r in paired
                          if r["fmt"] == fmt and (gdom is None or dom[r["ci"]] == gdom))
            dec = cnt["mat"] + cnt["std"]
            print(f"  paired-eq [{fmt:5s}]: mat {cnt['mat']/max(1,dec):.0%} of "
                  f"{dec} decisive | votes {dict(cnt)}")
    print("TRANSPLANT_DONE", flush=True)


if __name__ == "__main__":
    main()
