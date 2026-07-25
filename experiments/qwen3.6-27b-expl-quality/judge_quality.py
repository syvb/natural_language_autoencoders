"""LLM-judge the quality of NLA explanations (LOCAL, CPU + OpenRouter).

Two blind probes over data/manifest_quality.json + results/explanations_*.json:

  absolute (1 call per explanation, both arms):
    * CLAIMS — every distinct checkable claim enumerated and labeled
      SUPPORTED / PREDICTIVE / UNSUPPORTED / CONTRADICTED / META.
      Hallucination rate = (UNSUPPORTED+CONTRADICTED) / non-META claims.
    * COHERENCE 1-5 and USEFULNESS 1-5 on fixed rubrics.
  paired (1 call per context): both arms' explanations, A/B order randomized
    per context; preference on usefulness / groundedness / coherence / overall
    (TIE allowed).

The judge is never told which system produced which text, but it IS told how to
read both formats (prose vs coarse-to-fine list) — neither is "better" per se.

Judge model is SWAPPABLE: --judge or env JUDGE_MODEL (default
nex-agi/nex-n2-mini). Every judge gets its OWN cache and its OWN output file
(results/quality_<slug>.json), so re-running under a second judge never
clobbers or replays the first.

    JUDGE_WORKERS=400 python judge_quality.py
    python judge_quality.py --judge openai/gpt-4o-mini      # different judge
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "expl-quality-v1"
LABELS = ("SUPPORTED", "PREDICTIVE", "UNSUPPORTED", "CONTRADICTED", "META")
HALLUC = {"UNSUPPORTED", "CONTRADICTED"}

CLAIM_RE = re.compile(r"^\s*(?:\d+[.)]\s*)?(SUPPORTED|PREDICTIVE|UNSUPPORTED|"
                      r"CONTRADICTED|META)\s*::\s*(.+?)\s*$", re.M)
CLAIMS_BLOCK = re.compile(r"<claims>(.*?)</claims>", re.S | re.I)
COH_RE = re.compile(r"<coherence>\s*([1-5])\s*</coherence>", re.I)
USE_RE = re.compile(r"<usefulness>\s*([1-5])\s*</usefulness>", re.I)
PAIR_RE = {dim: re.compile(rf"<{dim}>\s*(A|B|TIE)\s*</{dim}>", re.I)
           for dim in ("more_useful", "more_grounded", "more_coherent", "overall")}

# ── shared guidance: what an NLA explanation is and how to read BOTH formats ──
GUIDANCE = """What you are reading: an interpretability tool read a language model's internal activation, recorded at the final token of a passage the model was processing, and wrote an "explanation" of that activation. A good explanation tells you what the model was reading (topic, specific content, entities, style) and what it was likely about to generate next.

How to read an explanation (important — the format may be unfamiliar):
- It may be PROSE (one or more sentences) or a LIST of short lines. Both are legitimate house styles of different tools. Never reward or penalize the format itself — judge only the content.
- If it is a list, the lines are ordered coarse-to-fine: the first line is the most general/most important summary, and each later line adds progressively finer detail. Later lines are allowed to be narrow or speculative details; that is the intended structure, not disorganization.
- Statements about what comes NEXT (e.g. "the model is about to list examples", predicted phrases or names) are claims about the upcoming text — check them against the true continuation, not the passage.
- Statements about the passage's genre, tone, register, structure, language, or the writer's intent are META observations; they can still be useful even though they assert no checkable fact.
- Fragments, sentence-completions or short quoted phrases (especially near the start) may be the tool restating or completing the passage's final words — treat them as claims about the passage's current text.
- The passage may be a plain document OR a chat transcript rendered with special role markers (e.g. <|im_start|>user … <|im_end|>). For a transcript, the "writer" is whichever party is mid-turn, and claims about the user's request, the assistant's reply, or the conversation as a whole are all checkable against the transcript."""

ABS_PROMPT = GUIDANCE + """

Here is the passage (the model had read exactly this much; the activation is at its final token):

<passage>
{passage}
</passage>

The text that actually came next (the model had NOT seen this yet, but the explanation may legitimately predict it):

<continuation>
{continuation}
</continuation>

The explanation to evaluate:

<explanation>
{expl}
</explanation>

Do three things, in order:

1. CLAIMS. List every distinct claim the explanation makes, one per line, each prefixed with exactly one label and " :: ":
   - SUPPORTED :: consistent with the passage.
   - PREDICTIVE :: about the upcoming text, and consistent with the true continuation.
   - UNSUPPORTED :: asserts a specific fact, entity, quotation, number, or event that appears nowhere in the passage or continuation and is not a reasonable inference or prediction from them.
   - CONTRADICTED :: conflicts with the passage or the continuation.
   - META :: only genre/tone/style/structure/language/intent — nothing concrete to check.
   Split compound statements into separate claims. Keep each claim's text short (a close paraphrase or quote of what the explanation says).

2. COHERENCE, 1-5: is the explanation well-formed writing on its own terms (ignore whether it is true)?
   1 = garbled, self-contradictory, or degenerate repetition; 2 = mostly disjointed or heavily redundant; 3 = readable but rambling, repetitive, or trailing off; 4 = clear and organized, minor flaws; 5 = crisp and well-structured throughout, no filler.
   (A coarse-to-fine list that reads as a clean progression scores high; do not mark a list down for being a list.)

3. USEFULNESS, 1-5: to an analyst who CANNOT see the passage and wants to know what the model was reading and about to write, how much would this explanation help? Weigh both informativeness and accuracy — wrong specifics actively mislead.
   1 = tells almost nothing, or is mostly misleading; 2 = generic gist only (could describe thousands of passages); 3 = correct topic plus a few accurate specifics; 4 = specific, mostly accurate picture of the passage; 5 = precise and accurate on both the passage's content and what comes next.

Answer in exactly this format:
<claims>
LABEL :: claim text
LABEL :: claim text
</claims>
<coherence>N</coherence>
<usefulness>N</usefulness>"""

PAIR_PROMPT = GUIDANCE + """

Here is the passage (the model had read exactly this much; the activation is at its final token):

<passage>
{passage}
</passage>

The text that actually came next (the model had NOT seen this yet, but an explanation may legitimately predict it):

<continuation>
{continuation}
</continuation>

Two different tools each explained the SAME activation:

<explanation_A>
{expl_a}
</explanation_A>

<explanation_B>
{expl_b}
</explanation_B>

Compare them on four dimensions (remember: judge content, never format):
- more_useful: which would better help an analyst who cannot see the passage know what the model was reading and about to write?
- more_grounded: which makes fewer unsupported or contradicted claims, relative to how much it says?
- more_coherent: which is better-formed writing on its own terms?
- overall: which explanation is better, all things considered?

Think briefly, then answer in exactly this format (A, B, or TIE for each):
<more_useful>A</more_useful>
<more_grounded>A</more_grounded>
<more_coherent>A</more_coherent>
<overall>A</overall>"""


def parse_abs(txt):
    m = CLAIMS_BLOCK.search(txt)
    coh, use = COH_RE.search(txt), USE_RE.search(txt)
    if not (m and coh and use):
        return None
    claims = [{"label": lab.upper(), "text": t}
              for lab, t in CLAIM_RE.findall(m.group(1))]
    return {"claims": claims, "coherence": int(coh.group(1)),
            "usefulness": int(use.group(1))}


def parse_pair(txt):
    out = {}
    for dim, rex in PAIR_RE.items():
        m = rex.search(txt)
        if not m:
            return None
        out[dim] = m.group(1).upper()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default=os.environ.get("JUDGE_MODEL",
                                                      "nex-agi/nex-n2-mini"))
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest_quality.json"))
    ap.add_argument("--explanations", nargs="+",
                    default=[str(HERE / "results" / "explanations_mat.json"),
                             str(HERE / "results" / "explanations_std.json")])
    ap.add_argument("--skip-paired", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N contexts")
    ap.add_argument("--seed", type=int, default=0, help="paired A/B order seed")
    args = ap.parse_args()

    judge = args.judge
    slug = re.sub(r"[^a-z0-9]+", "-", judge.lower()).strip("-")
    cache_path = HERE / "results" / f".quality_cache_{slug}.json"
    out_path = HERE / "results" / f"quality_{slug}.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    print(f"[judge] {judge} | cache {len(cache)} entries", flush=True)

    def ask(job):
        """job = (kind, prompt). Returns (key, parsed-or-None). Raw text is
        cached so parse logic can be improved without re-paying the calls."""
        kind, prompt = job
        k = hashlib.sha1((PROMPT_VERSION + "|" + kind + "|" + prompt).encode()).hexdigest()
        parse = parse_abs if kind == "abs" else parse_pair
        if cache.get(k) is not None:
            return k, parse(cache[k])
        body = {"model": judge, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        # 300s timeout + backoff: nex-n2-mini often reasons for >120s
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    URL, json.dumps(body).encode(),
                    {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=300))
                txt = r["choices"][0]["message"]["content"]
                if parse(txt) is not None:
                    cache[k] = txt
                    return k, parse(txt)
            except Exception:
                time.sleep(5 * (attempt + 1))
        return k, None

    def run_jobs(jobs, tag):
        workers = int(os.environ.get("JUDGE_WORKERS", "400"))
        fails, res = 0, []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(ask, j) for j in jobs]
            for n, f in enumerate(futs, 1):
                kk, v = f.result()
                res.append(v)
                if v is None:
                    fails += 1
                if n % 200 == 0:
                    json.dump(cache, open(cache_path, "w"))
                    print(f"[{tag}] {n}/{len(jobs)} ({fails} unparsed)", flush=True)
        json.dump(cache, open(cache_path, "w"))
        print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
        return res

    man = {c["ci"]: c for c in json.load(open(args.manifest))["contexts"]}
    arms = {}
    for ef in args.explanations:
        data = json.load(open(ef))
        arms[data["meta"]["model"]] = {e["ci"]: e for e in data["entries"]}
    cis = sorted(set.intersection(*(set(v) for v in arms.values())) & set(man))
    if args.limit:
        cis = cis[: args.limit]
    print(f"[data] {len(cis)} contexts x {len(arms)} arms", flush=True)

    def body_of(arm, ci):
        return "\n".join(arms[arm][ci]["explanations"][0])

    report = {"meta": {"judge": judge, "prompt_version": PROMPT_VERSION,
                       "n_contexts": len(cis), "arms": sorted(arms)}}

    # ── absolute leg ─────────────────────────────────────────────────────────
    jobs, idx = [], []
    for arm in sorted(arms):
        for ci in cis:
            jobs.append(("abs", ABS_PROMPT.format(
                passage=man[ci]["prefix_text"],
                continuation=man[ci]["continuation_text"],
                expl=body_of(arm, ci))))
            idx.append((arm, ci))
    res = run_jobs(jobs, "absolute")
    report["absolute"] = [
        {"arm": arm, "ci": ci, **v} for (arm, ci), v in zip(idx, res) if v]

    # ── paired leg ───────────────────────────────────────────────────────────
    if not args.skip_paired and len(arms) == 2:
        a1, a2 = sorted(arms)   # mat, std
        rng = random.Random(args.seed)
        order = {ci: rng.random() < 0.5 for ci in cis}   # True => A=a1
        jobs = []
        for ci in cis:
            fa, fb = (a1, a2) if order[ci] else (a2, a1)
            jobs.append(("pair", PAIR_PROMPT.format(
                passage=man[ci]["prefix_text"],
                continuation=man[ci]["continuation_text"],
                expl_a=body_of(fa, ci), expl_b=body_of(fb, ci))))
        res = run_jobs(jobs, "paired")
        paired = []
        for ci, v in zip(cis, res):
            if not v:
                continue
            # translate A/B back to arm names
            rec = {"ci": ci}
            for dim, ab in v.items():
                rec[dim] = ("TIE" if ab == "TIE"
                            else (a1 if (ab == "A") == order[ci] else a2))
            paired.append(rec)
        report["paired"] = paired

    out_path.write_text(json.dumps(report))
    print(f"[write] {out_path}", flush=True)

    # ── quick console summary (overall + per domain) ─────────────────────────
    dom = {ci: man[ci].get("domain", "pretrain") for ci in cis}
    groups = [("all", None)] + [(d, d) for d in sorted(set(dom.values()))]
    for gname, gdom in groups:
      for arm in sorted(arms):
        rows = [r for r in report["absolute"] if r["arm"] == arm
                and (gdom is None or dom[r["ci"]] == gdom)]
        if not rows:
            continue
        ncl = [len([c for c in r["claims"] if c["label"] != "META"]) for r in rows]
        hal = [len([c for c in r["claims"] if c["label"] in HALLUC]) for r in rows]
        rate = sum(hal) / max(1, sum(ncl))
        coh = sum(r["coherence"] for r in rows) / len(rows)
        use = sum(r["usefulness"] for r in rows) / len(rows)
        print(f"[{gname}] {arm}: n={len(rows)} | halluc {sum(hal)}/{sum(ncl)} = "
              f"{rate:.1%} of checkable claims | coherence {coh:.2f} | "
              f"usefulness {use:.2f}", flush=True)
    if "paired" in report:
        from collections import Counter
        for gname, gdom in groups:
            rows = [r for r in report["paired"]
                    if gdom is None or dom[r["ci"]] == gdom]
            for dim in PAIR_RE:
                cnt = Counter(r[dim] for r in rows)
                print(f"[{gname}] paired {dim}: {dict(cnt)}", flush=True)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
