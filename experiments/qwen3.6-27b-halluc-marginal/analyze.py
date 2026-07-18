"""Mine negative-marginal matryoshka items, judge faithfulness, and compare
against the standard NLA's corresponding sentences (LOCAL, CPU + OpenRouter).

Inputs (produced by score_subsets.py on the GPU box):
  results/subset_scores_mat.json   items = matryoshka lines
  results/subset_scores_std.json   items = std <explanation> sentence units
plus the suffix-eval manifest for source passages and true continuations.

Stages (idempotent; all judge calls cached per-model):
  1. marginals: m[0] = pfx[0], m[k] = pfx[k] - pfx[k-1] for every rollout.
  2. mine: matryoshka items at position <3 with m < 0 (the critic says the
     item made reconstruction WORSE — candidate critic-detected hallucination).
  3. faithfulness judge: EVERY top-3 mat item and EVERY std unit classified
     against the source passage + true continuation:
       SUPPORTED / CONTRADICTED / FABRICATED / META
     (hallucination = CONTRADICTED or FABRICATED).
  4. match judge: for each mined case, find the std sentence (same context,
     each of the 4 std rollouts) expressing the corresponding claim, if any.
  5. report: results/analysis.json + results/cases.md (top case studies).

Judge = nex-agi/nex-n2-mini on OpenRouter (env JUDGE_MODEL to override).
"""
import argparse
import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SE = HERE.parent / "qwen3.6-27b-suffix-eval"
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE = os.environ.get("JUDGE_MODEL", "nex-agi/nex-n2-mini")
_slug = re.sub(r"[^a-z0-9]+", "-", JUDGE.lower())
CACHE_PATH = HERE / "results" / f".judge_cache_{_slug}.json"
PROMPT_VERSION = "halluc-marginal-v1"
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")
NEG_CLEAR = -0.01   # "clearly negative" tier

FAITH_PROMPT = """You are auditing an interpretability tool. The tool read a language model's internal state while the model was reading a passage, and produced a series of short notes describing the passage and the model's state. You will fact-check ONE note.

Here is the passage (the model had read exactly this much):

<passage>
{passage}
</passage>

The text that actually came next (the model had NOT seen this yet, but the note may legitimately predict it):

<continuation>
{continuation}
</continuation>

The note to fact-check:

<note>
{note}
</note>

Classify the note:
- SUPPORTED: its claims are consistent with the passage (or it predicts something consistent with the actual continuation).
- CONTRADICTED: it misstates something — conflicts with facts, entities, numbers, or events in the passage/continuation.
- FABRICATED: it introduces specific entities, facts, or events that appear nowhere in the passage/continuation and are not reasonable inferences.
- META: it only describes genre, tone, style, structure, language, or the writer's intent — nothing concrete to fact-check.

Think briefly, then answer with exactly one word in <answer> tags, e.g. <answer>SUPPORTED</answer>."""

MATCH_PROMPT = """Two different tools each summarized the same passage-reading state of a language model. Tool A produced a (possibly wrong) note. You must decide whether Tool B's summary makes the same claim anywhere.

Tool A's note:

<note>
{note}
</note>

Tool B's summary, as numbered sentences:

{sentences}

Does any of Tool B's sentences express substantially the same claim as Tool A's note (same fact, entity, event, or prediction — wording may differ)? If yes, answer with the number of the best-matching sentence. If none of them makes that claim, answer NONE.

Think briefly, then answer in <answer> tags, e.g. <answer>2</answer> or <answer>NONE</answer>."""

FAITH_RE = re.compile(r"<answer>\s*(SUPPORTED|CONTRADICTED|FABRICATED|META)\s*</answer>", re.I)
MATCH_RE = re.compile(r"<answer>\s*(NONE|\d+)\s*</answer>", re.I)

cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}


def ask(prompt, rex):
    k = hashlib.sha1((PROMPT_VERSION + "|" + JUDGE + "|" + prompt).encode()).hexdigest()
    if k in cache and cache[k] is not None:
        return k, cache[k]
    body = {"model": JUDGE, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                URL, json.dumps(body).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            txt = r["choices"][0]["message"]["content"]
            m = rex.search(txt)
            if m:
                return k, m.group(1).upper()
        except Exception:
            continue
    return k, None


def run_jobs(jobs, rex, tag):
    """jobs: list of prompts. Returns list of verdicts (None on failure).
    Cache is checkpointed every chunk so a crash loses <=500 calls."""
    CACHE_PATH.parent.mkdir(exist_ok=True)
    res = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "24"))) as ex:
        for c0 in range(0, len(jobs), 500):
            res += list(ex.map(lambda p: ask(p, rex), jobs[c0: c0 + 500]))
            for k, v in res[c0:]:
                cache[k] = v
            json.dump(cache, open(CACHE_PATH, "w"))
            print(f"  [{tag}] {min(c0 + 500, len(jobs))}/{len(jobs)}", flush=True)
    fails = sum(1 for _, v in res if v is None)
    print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
    return [v for _, v in res]


def marginals(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def load_all():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    mat = json.load(open(HERE / "results" / "subset_scores_mat.json"))["entries"]
    std = json.load(open(HERE / "results" / "subset_scores_std.json"))["entries"]
    return man, mat, std


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-cases", type=int, default=20)
    args = ap.parse_args()
    man, mat, std = load_all()

    # ── stage 3: faithfulness sweep (mat top-3 items + ALL std units) ────────
    faith_items = []   # (arm, ci, ri, k, text)
    for arm, entries, kmax in (("mat", mat, 3), ("std", std, 99)):
        for e in entries:
            for ri, rec in enumerate(e["rollouts"]):
                if not rec:
                    continue
                for k, u in enumerate(rec["units"][:kmax]):
                    faith_items.append((arm, e["ci"], ri, k, u))
    prompts = [FAITH_PROMPT.format(passage=man[ci]["prefix_text"],
                                   continuation=man[ci]["answer_text"], note=u)
               for _, ci, _, _, u in faith_items]
    verdicts = run_jobs(prompts, FAITH_RE, "faithfulness")
    faith = {}
    for (arm, ci, ri, k, u), v in zip(faith_items, verdicts):
        faith[(arm, ci, ri, k)] = v
    json.dump({f"{a}|{c}|{r}|{k}": v for (a, c, r, k), v in faith.items()},
              open(HERE / "results" / "judged_faithfulness.json", "w"))

    # ── stages 1-2: marginals + mining ───────────────────────────────────────
    mined = []
    for e in mat:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            m = marginals(rec)
            for k in range(min(3, len(m))):
                if m[k] < 0:
                    dmg = rec["full"] - rec["loo"][k] if rec["loo"] else None
                    mined.append({
                        "ci": e["ci"], "ri": ri, "k": k,
                        "item": rec["units"][k], "marginal": round(m[k], 5),
                        "solo": rec["solo"][k],
                        "loo_damage": round(dmg, 5) if dmg is not None else None,
                        "full": rec["full"], "cjk": bool(CJK.search(rec["units"][k])),
                        "verdict": faith.get(("mat", e["ci"], ri, k)),
                    })
    mined.sort(key=lambda c: c["marginal"])
    print(f"[mine] {len(mined)} negative top-3 mat items "
          f"({sum(1 for c in mined if c['marginal'] <= NEG_CLEAR)} clearly negative, "
          f"{len({c['ci'] for c in mined})} distinct contexts, "
          f"{sum(1 for c in mined if c['cjk'])} CJK-tainted)", flush=True)

    # ── stage 4: match each mined item against the 4 std rollouts ────────────
    std_by_ci = {e["ci"]: e for e in std}
    match_jobs, match_meta = [], []
    for mi, c in enumerate(mined):
        se_e = std_by_ci.get(c["ci"])
        if not se_e:
            continue
        for ri, rec in enumerate(se_e["rollouts"]):
            if not rec:
                continue
            sents = "\n".join(f"{i+1}. {u.strip()}" for i, u in enumerate(rec["units"]))
            match_jobs.append(MATCH_PROMPT.format(note=c["item"], sentences=sents))
            match_meta.append((mi, ri))
    mverd = run_jobs(match_jobs, MATCH_RE, "match")
    for c in mined:
        c["std_matches"] = []
    for (mi, ri), v in zip(match_meta, mverd):
        if v is None or v == "NONE":
            continue
        c = mined[mi]
        rec = std_by_ci[c["ci"]]["rollouts"][ri]
        k = int(v) - 1
        if not (0 <= k < len(rec["units"])):
            continue
        m = marginals(rec)
        dmg = rec["full"] - rec["loo"][k] if rec["loo"] else None
        c["std_matches"].append({
            "ri": ri, "k": k, "sentence": rec["units"][k],
            "marginal": round(m[k], 5), "solo": rec["solo"][k],
            "loo_damage": round(dmg, 5) if dmg is not None else None,
            "verdict": faith.get(("std", c["ci"], ri, k)),
        })

    # ── stage 5: aggregate + report ──────────────────────────────────────────
    def faith_table(arm, entries, kmax):
        rows = []
        for e in entries:
            for ri, rec in enumerate(e["rollouts"]):
                if not rec:
                    continue
                m = marginals(rec)
                for k in range(min(kmax, len(m))):
                    v = faith.get((arm, e["ci"], ri, k))
                    if v:
                        rows.append((m[k], v))
        return rows

    summary = {"judge": JUDGE, "n_mined": len(mined),
               "n_mined_clear": sum(1 for c in mined if c["marginal"] <= NEG_CLEAR),
               "mined_contexts": len({c["ci"] for c in mined})}
    for arm, entries, kmax in (("mat", mat, 3), ("std", std, 99)):
        rows = faith_table(arm, entries, kmax)
        neg = [v for m, v in rows if m < 0]
        pos = [v for m, v in rows if m >= 0]
        halluc = {"CONTRADICTED", "FABRICATED"}
        summary[arm] = {
            "n_items": len(rows),
            "verdicts": dict(Counter(v for _, v in rows)),
            "neg_marginal_frac": round(np.mean([m < 0 for m, _ in rows]), 4),
            "p_halluc_given_neg": round(np.mean([v in halluc for v in neg]), 4) if neg else None,
            "p_halluc_given_pos": round(np.mean([v in halluc for v in pos]), 4) if pos else None,
            "p_neg_given_halluc": round(np.mean(
                [m < 0 for m, v in rows if v in halluc]), 4) if any(
                v in halluc for _, v in rows) else None,
            "p_neg_given_supported": round(np.mean(
                [m < 0 for m, v in rows if v == "SUPPORTED"]), 4) if any(
                v == "SUPPORTED" for _, v in rows) else None,
        }
    matched = [c for c in mined if c["std_matches"]]
    summary["match"] = {
        "mined_with_std_match": len(matched),
        "mined_without_std_match": len(mined) - len(matched),
        "matched_std_marginal_mean": round(float(np.mean(
            [m["marginal"] for c in matched for m in c["std_matches"]])), 4) if matched else None,
        "matched_std_marginal_neg_frac": round(float(np.mean(
            [m["marginal"] < 0 for c in matched for m in c["std_matches"]])), 4) if matched else None,
        "matched_std_halluc_frac": round(float(np.mean(
            [m["verdict"] in ("CONTRADICTED", "FABRICATED")
             for c in matched for m in c["std_matches"] if m["verdict"]])), 4) if matched else None,
    }
    json.dump({"summary": summary, "mined": mined},
              open(HERE / "results" / "analysis.json", "w"), indent=1)
    print(json.dumps(summary, indent=2), flush=True)

    # case studies: strongest negatives, skipping CJK-tainted for readability
    lines = ["# Negative-marginal matryoshka items vs the standard NLA",
             "", f"Judge: `{JUDGE}`. Marginal of item k = FVE(items 1..k) − "
             "FVE(items 1..k−1), own-critic. Damage = full − leave-one-out.", ""]
    shown = 0
    for c in mined:
        if shown >= args.top_cases:
            break
        if c["cjk"]:
            continue
        ctx = man[c["ci"]]
        e_mat = next(e for e in mat if e["ci"] == c["ci"])
        rec = e_mat["rollouts"][c["ri"]]
        lines += [f"## Case {shown+1}: ci={c['ci']} rollout={c['ri']} "
                  f"item {c['k']+1} — marginal {c['marginal']:+.3f}",
                  "", f"**Source tail** (…{ctx['prefix_text'][-300:]!r})",
                  f"**True continuation:** {ctx['answer_text']!r}", "",
                  "**Matryoshka explanation** (→ = the offending item):", ""]
        m = marginals(rec)
        for k, u in enumerate(rec["units"]):
            mark = "→" if k == c["k"] else " "
            lines.append(f"- {mark} [{m[k]:+.3f}] {u}")
        lines += ["", f"Item verdict: **{c['verdict']}** | solo {c['solo']:+.3f}"
                  + (f" | LOO damage {c['loo_damage']:+.3f}" if c["loo_damage"] is not None else ""), ""]
        if c["std_matches"]:
            for sm in c["std_matches"]:
                lines.append(f"**Std match** (rollout {sm['ri']}, sent {sm['k']+1}, "
                             f"marginal {sm['marginal']:+.3f}, solo {sm['solo']:+.3f}, "
                             f"verdict {sm['verdict']}): {sm['sentence'].strip()}")
        else:
            lines.append("**Std match:** none — the standard NLA does not assert this claim.")
        lines.append("")
        shown += 1
    (HERE / "results" / "cases.md").write_text("\n".join(lines))
    print(f"[saved] results/analysis.json, results/cases.md "
          f"({shown} case studies)", flush=True)


if __name__ == "__main__":
    main()
