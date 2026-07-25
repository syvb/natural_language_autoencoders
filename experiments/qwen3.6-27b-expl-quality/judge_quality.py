"""LLM-judge the quality of NLA explanations (LOCAL, CPU + OpenRouter).

Probes over data/manifest_quality.json + results/explanations_*.json:

  absolute (1 call per explanation): every distinct claim enumerated and
    labeled SUPPORTED / PREDICTIVE / PREDICTIVE_UNVERIFIABLE / UNSUPPORTED /
    CONTRADICTED / META / RESTATE, plus COHERENCE, USEFULNESS and
    INFORMATIVENESS 1-5 on fixed rubrics.
    Hallucination rate = (UNSUPPORTED+CONTRADICTED) / checkable claims, where
    checkable = SUPPORTED+PREDICTIVE+UNSUPPORTED+CONTRADICTED. (RESTATE, META
    and PREDICTIVE_UNVERIFIABLE are excluded from the denominator and
    reported separately.)
  calibration arms through the same absolute pipeline:
    skyline    — the passage's own tail as the "explanation": the judge's
                 hallucination-rate floor / usefulness ceiling.
    floor_mat / floor_std — within-domain derangement (context i judged
                 against context i+1's explanation): the judge's sensitivity
                 ceiling / usefulness floor.
  paired (2 calls per context, BOTH A/B orders): preference on usefulness /
    groundedness / coherence / overall (TIE allowed). Both orders let the
    analysis report position-consistent wins and the order-flip rate.

The judge is never told which system produced which text, but it IS told how
to read both formats (prose vs coarse-to-fine list) — neither is "better".

Judge model is SWAPPABLE: --judge or env JUDGE_MODEL (default
nex-agi/nex-n2-mini). Every judge gets its OWN cache and its OWN output file
(results/quality_<slug>.json), so re-running under a second judge never
clobbers or replays the first. Raw responses (parseable or not) are cached, so
parse logic can be improved without re-paying calls.

    JUDGE_WORKERS=400 python judge_quality.py
    python judge_quality.py --judge openai/gpt-4o-mini      # different judge
    python judge_quality.py --limit 20                      # stratified smoke
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "expl-quality-v2"
LABELS = ("SUPPORTED", "PREDICTIVE", "PREDICTIVE_UNVERIFIABLE",
          "UNSUPPORTED", "CONTRADICTED", "META", "RESTATE")
HALLUC = {"UNSUPPORTED", "CONTRADICTED"}
CHECKABLE = {"SUPPORTED", "PREDICTIVE", "UNSUPPORTED", "CONTRADICTED"}

# PREDICTIVE_UNVERIFIABLE before PREDICTIVE — the alternation is first-match
_LAB = "|".join(sorted(LABELS, key=len, reverse=True))
CLAIM_RE = re.compile(rf"^\s*(?:(?:\d+[.)]|[-*•])\s*)*({_LAB})\s*::\s*(.+?)\s*$",
                      re.M | re.I)
CLAIMS_BLOCK = re.compile(r"<claims>(.*?)</claims>", re.S | re.I)
SCORE_RE = {dim: re.compile(rf"<{dim}>\s*([1-5])\s*</{dim}>", re.I)
            for dim in ("coherence", "usefulness", "informativeness")}
PAIR_RE = {dim: re.compile(rf"<{dim}>\s*(A|B|TIE)\s*</{dim}>", re.I)
           for dim in ("more_useful", "more_grounded", "more_coherent", "overall")}

# ── shared guidance: what an NLA explanation is and how to read BOTH formats ──
GUIDANCE = """What you are reading: an interpretability tool read a language model's internal activation, recorded at the final token of a passage the model was processing, and wrote an "explanation" of that activation. A good explanation tells you what the model was reading (topic, specific content, entities, style) and what it was likely about to generate next.

How to read an explanation (important — the format may be unfamiliar):
- It may be PROSE (one or more sentences) or a LIST of short lines. Both are legitimate house styles of different tools. Never reward or penalize the format itself — judge only the content.
- If it is a list, the lines are ordered coarse-to-fine: the first line is the most general summary, and each later line adds progressively finer detail. If it is prose, it may open with broad framing (genre, register, topic) before specifics. Neither structure is disorganization.
- The coarse-to-fine license is about ORGANIZATION only, never evidence: label the factual accuracy of a late, narrow, or speculative line exactly as strictly as any other claim.
- Statements about what comes NEXT (e.g. "the model is about to list examples", predicted phrases or names) are claims about the upcoming text — check them against the true continuation, not the passage.
- Fragments, sentence-completions or short quoted phrases (especially near the start) may be the tool restating or completing the passage's final words — these carry little information and get the RESTATE label.
- The passage may be a plain document OR a chat transcript rendered with special role markers (e.g. <|im_start|>user … <|im_end|>). For a transcript, the "writer" is whichever party is mid-turn, and claims about the user's request, the assistant's reply, or the conversation as a whole are all checkable against the transcript. Note a transcript's continuation may be another party's turn, which no tool could predict."""

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

Do four things, in order:

1. CLAIMS. Split the explanation into the SMALLEST independently checkable units — one claim per line, splitting every compound statement — each prefixed with exactly one label and " :: ":
   - SUPPORTED :: consistent with the passage.
   - PREDICTIVE :: about the upcoming text, and the true continuation confirms it or at least does not conflict with it.
   - PREDICTIVE_UNVERIFIABLE :: about the upcoming text, but the shown continuation cannot resolve it either way (too short, or the next turn belongs to a different speaker).
   - UNSUPPORTED :: asserts a specific fact, entity, quotation, number, or event that appears nowhere in the passage or continuation and is not a reasonable inference from them.
   - CONTRADICTED :: conflicts with the passage or the continuation — including a prediction the continuation shows to be wrong, and a genre/tone/language/intent characterization that is clearly wrong for this passage.
   - META :: a genre/tone/style/structure/language/intent observation that is not verifiably wrong — nothing concrete to check.
   - RESTATE :: merely restates, quotes, or completes the passage's final words without adding information beyond them.
   Keep each claim's text short (a close paraphrase or quote of what the explanation says).

2. COHERENCE, 1-5: is the explanation well-formed writing on its own terms (ignore whether it is true)?
   1 = garbled, self-contradictory, or degenerate repetition; 2 = mostly disjointed or heavily redundant; 3 = readable but rambling, repetitive, or trailing off; 4 = clear and organized, minor flaws; 5 = crisp and well-structured throughout, no filler.
   (A clean coarse-to-fine list scores high, and so does clean prose — judge the writing, not the format.)

3. USEFULNESS, 1-5: to an analyst who CANNOT see the passage and wants to know what the model was reading and about to write, how much would this explanation help? Weigh both informativeness and accuracy — wrong specifics actively mislead.
   1 = tells almost nothing, or is mostly misleading; 2 = generic gist only (could describe thousands of passages); 3 = correct topic plus a few accurate specifics; 4 = specific, mostly accurate picture of the passage; 5 = precise and accurate on both the passage's content and what comes next.

4. INFORMATIVENESS, 1-5: how specific a picture does it paint of this PARTICULAR passage, regardless of accuracy?
   1 = could describe thousands of passages; 3 = narrows to a topic and situation; 5 = pins down this passage precisely (entities, specifics, position in the text).

Answer in exactly this format — claims lines are "LABEL :: text" with no bullets or extra prose:
<claims>
LABEL :: claim text
LABEL :: claim text
</claims>
<coherence>N</coherence>
<usefulness>N</usefulness>
<informativeness>N</informativeness>"""

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

NUDGE = {
    "abs": "\n\nIMPORTANT: end your reply with the literal output format exactly "
           "as specified — a <claims> block containing ONLY lines of the form "
           "'LABEL :: text' (no bullets, no numbering, uppercase label), then "
           "<coherence>N</coherence>, <usefulness>N</usefulness>, "
           "<informativeness>N</informativeness>.",
    "pair": "\n\nIMPORTANT: end your reply with exactly four tags, each A, B or "
            "TIE: <more_useful></more_useful> <more_grounded></more_grounded> "
            "<more_coherent></more_coherent> <overall></overall>.",
}


def parse_abs(txt):
    m = CLAIMS_BLOCK.search(txt)
    if not m:
        return None
    scores = {}
    for dim, rex in SCORE_RE.items():
        s = rex.search(txt)
        if not s:
            return None
        scores[dim] = int(s.group(1))
    block = m.group(1)
    claims = [{"label": lab.upper(), "text": t}
              for lab, t in CLAIM_RE.findall(block)]
    # a block with non-blank lines that did NOT parse as claims means the
    # model drifted from the format — treat as a parse failure so retry fires
    nonblank = [ln for ln in block.splitlines() if ln.strip()]
    if len(claims) < len(nonblank):
        return None
    return {"claims": claims, **scores}


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
    ap.add_argument("--skip-controls", action="store_true",
                    help="skip the skyline + derangement-floor calibration arms")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: first N contexts, stratified across domains")
    ap.add_argument("--seed", type=int, default=0, help="paired A/B order seed")
    args = ap.parse_args()

    judge = args.judge
    slug = re.sub(r"[^a-z0-9]+", "-", judge.lower()).strip("-")
    cache_path = HERE / "results" / f".quality_cache_{slug}.json"
    out_path = HERE / "results" / f"quality_{slug}.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    print(f"[judge] {judge} | cache {len(cache)} entries", flush=True)

    def flush_cache():
        tmp = cache_path.with_suffix(".tmp")
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, cache_path)

    def ask(job):
        """job = (kind, prompt). Returns (key, raw_text_or_None, parsed_or_None).
        NEVER touches `cache` (the main thread owns it — a json.dump racing
        worker inserts can corrupt the file)."""
        kind, prompt = job
        k = hashlib.sha1((PROMPT_VERSION + "|" + kind + "|" + prompt).encode()).hexdigest()
        parse = parse_abs if kind == "abs" else parse_pair
        if cache.get(k) is not None:
            return k, None, parse(cache[k])
        txt = None
        # 300s timeout + backoff: nex-n2-mini often reasons for >120s
        for attempt in range(3):
            p = prompt + NUDGE[kind] if attempt else prompt
            body = {"model": judge, "temperature": 0,
                    "messages": [{"role": "user", "content": p}]}
            try:
                req = urllib.request.Request(
                    URL, json.dumps(body).encode(),
                    {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=300))
                txt = r["choices"][0]["message"]["content"]
                if parse(txt) is not None:
                    break
            except Exception:
                time.sleep(5 * (attempt + 1))
        # raw text cached (by the caller) even when unparseable, so improved
        # parse logic can rescue it later without re-paying the call
        return k, txt, (parse(txt) if txt is not None else None)

    def run_jobs(jobs, tag):
        workers = int(os.environ.get("JUDGE_WORKERS", "400"))
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
                if n % 200 == 0:
                    flush_cache()
                    print(f"[{tag}] {n}/{len(jobs)} ({fails} unparsed)", flush=True)
        flush_cache()
        print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
        return res

    man = {c["ci"]: c for c in json.load(open(args.manifest))["contexts"]}
    arms = {}
    for ef in args.explanations:
        data = json.load(open(ef))
        arms[data["meta"]["model"]] = {e["ci"]: e for e in data["entries"]}
    cis = sorted(set.intersection(*(set(v) for v in arms.values())) & set(man))
    dom = {ci: man[ci].get("domain", "pretrain") for ci in cis}
    if args.limit:
        # stratified: round-robin across domains so a smoke run exercises both
        by_dom = {}
        for ci in cis:
            by_dom.setdefault(dom[ci], []).append(ci)
        picked, i = [], 0
        while len(picked) < args.limit and any(by_dom.values()):
            for d in sorted(by_dom):
                if by_dom[d] and len(picked) < args.limit:
                    picked.append(by_dom[d].pop(0))
            i += 1
        cis = sorted(picked)
    print(f"[data] {len(cis)} contexts x {len(arms)} arms "
          f"({Counter(dom[ci] for ci in cis)})", flush=True)

    def body_of(arm, ci):
        return "\n".join(arms[arm][ci]["explanations"][0])

    def abs_job(ci, body):
        return ("abs", ABS_PROMPT.format(
            passage=man[ci]["prefix_text"],
            continuation=man[ci]["continuation_text"], expl=body))

    report = {"meta": {"judge": judge, "prompt_version": PROMPT_VERSION,
                       "n_contexts": len(cis), "arms": sorted(arms)}}

    # ── absolute legs: real arms + calibration arms ──────────────────────────
    # within-domain derangement donors for the floor arms
    by_dom = {}
    for ci in cis:
        by_dom.setdefault(dom[ci], []).append(ci)
    donor = {}
    for d, group in by_dom.items():
        for i, ci in enumerate(group):
            donor[ci] = group[(i + 1) % len(group)]

    legs = [(arm, arm, lambda ci, a=arm: body_of(a, ci)) for arm in sorted(arms)]
    if not args.skip_controls:
        legs.append(("skyline", None,
                     lambda ci: " ".join(man[ci]["prefix_text"].split()[-150:])))
        for arm in sorted(arms):
            legs.append((f"floor_{arm}", None,
                         lambda ci, a=arm: body_of(a, donor[ci])))

    jobs, idx = [], []
    for leg_name, _, body_fn in legs:
        for ci in cis:
            jobs.append(abs_job(ci, body_fn(ci)))
            idx.append((leg_name, ci))
    res = run_jobs(jobs, "absolute")
    report["absolute"] = [
        {"arm": leg, "ci": ci, **v} for (leg, ci), v in zip(idx, res) if v]

    # ── paired leg: BOTH A/B orders per context ──────────────────────────────
    if not args.skip_paired and len(arms) == 2:
        a1, a2 = sorted(arms)   # mat, std
        jobs, meta = [], []
        for ci in cis:
            for order, (fa, fb) in [("ms", (a1, a2)), ("sm", (a2, a1))]:
                jobs.append(("pair", PAIR_PROMPT.format(
                    passage=man[ci]["prefix_text"],
                    continuation=man[ci]["continuation_text"],
                    expl_a=body_of(fa, ci), expl_b=body_of(fb, ci))))
                meta.append((ci, order, fa, fb))
        res = run_jobs(jobs, "paired")
        paired = []
        for (ci, order, fa, fb), v in zip(meta, res):
            if not v:
                continue
            rec = {"ci": ci, "order": order}
            for dim, ab in v.items():
                rec[dim] = "TIE" if ab == "TIE" else (fa if ab == "A" else fb)
            paired.append(rec)
        report["paired"] = paired

    out_path.write_text(json.dumps(report))
    print(f"[write] {out_path}", flush=True)

    # ── quick console summary (overall + per domain) ─────────────────────────
    groups = [("all", None)] + [(d, d) for d in sorted(set(dom.values()))]
    leg_names = [l for l, _, _ in legs]
    for gname, gdom in groups:
        for leg in leg_names:
            rows = [r for r in report["absolute"] if r["arm"] == leg
                    and (gdom is None or dom[r["ci"]] == gdom)]
            if not rows:
                continue
            chk = sum(len([c for c in r["claims"] if c["label"] in CHECKABLE])
                      for r in rows)
            hal = sum(len([c for c in r["claims"] if c["label"] in HALLUC])
                      for r in rows)
            any_h = sum(any(c["label"] in HALLUC for c in r["claims"])
                        for r in rows) / len(rows)
            coh = sum(r["coherence"] for r in rows) / len(rows)
            use = sum(r["usefulness"] for r in rows) / len(rows)
            inf = sum(r["informativeness"] for r in rows) / len(rows)
            print(f"[{gname}] {leg}: n={len(rows)} | halluc {hal}/{chk} = "
                  f"{hal/max(1,chk):.1%} of checkable | P(>=1 halluc) {any_h:.0%} | "
                  f"coh {coh:.2f} use {use:.2f} info {inf:.2f}", flush=True)
    if "paired" in report:
        for gname, gdom in groups:
            rows = [r for r in report["paired"]
                    if gdom is None or dom[r["ci"]] == gdom]
            for dim in PAIR_RE:
                cnt = Counter(r[dim] for r in rows)
                print(f"[{gname}] paired {dim}: {dict(cnt)}", flush=True)
    print("JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
