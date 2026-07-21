"""Suffix Prediction with the FINAL TOKEN revealed to the judge (same-doc eval).

The matryoshka's first lines often literally restate/complete the passage's
final token — a trivially front-loadable cue. Here the judge is TOLD that token
outright, so any surviving low-budget advantage is semantic front-loading
beyond the last token. Three legs (all against the same-document option sets):

  tokonly : final token + options, NO explanation  (baseline floor, 250 calls)
  full+ft : full explanation + final token, both arms, 2 rollouts
  budget+ft : first-N-token truncations + final token, both arms, 1 rollout

Shares the Haiku cache (kinds "fttok"/"ftexpl" keep keys distinct).

    /home/debian/nanoNLA-multi-input/.venv-cpu/bin/python grade_suffix_ft.py
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from grade_suffix import (KEY, URL, GRADER, ANS_RE, LETTERS, wilson,
                          cache, CACHE_PATH)

HERE = Path(__file__).resolve().parent
FT_VERSION = "nla-suffix-ft-v1"


def ask_raw(job):
    """job = (kind, full_prompt). The prompt is prebuilt — no outer template."""
    kind, prompt = job[0], job[1]
    k = hashlib.sha1((FT_VERSION + "|" + kind + "|" + prompt).encode()).hexdigest()
    if k in cache and cache[k] is not None:
        return k, cache[k]
    body_json = {"model": GRADER, "temperature": 0,
                 "messages": [{"role": "user", "content": prompt}]}
    # 300s timeout: nex-n2-mini often reasons for >120s on these prompts —
    # a shorter timeout silently burns all 3 attempts on calls that would succeed
    for attempt in range(3):
        if attempt == 2:
            body_json["messages"][0]["content"] = prompt + \
                "\n\nRespond with exactly one letter in <answer> tags."
        try:
            req = urllib.request.Request(
                URL, json.dumps(body_json).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            m = ANS_RE.search(r["choices"][0]["message"]["content"])
            if m:
                return k, LETTERS.index(m.group(1).upper())
        except Exception:
            time.sleep(5 * (attempt + 1))
            continue
    return k, None


def run_jobs(jobs, tag):
    workers = int(os.environ.get("SUFFIX_WORKERS", "64"))
    fails, res = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(ask_raw, j) for j in jobs]
        for n, f in enumerate(futs, 1):
            kk, v = f.result()
            cache[kk] = v
            res.append(v)
            if v is None:
                fails += 1
            if n % 200 == 0:
                json.dump(cache, open(CACHE_PATH, "w"))
                print(f"[{tag}] {n}/{len(jobs)} ({fails} unparsed)", flush=True)
    json.dump(cache, open(CACHE_PATH, "w"))
    print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
    return res

FT_EXPL_PROMPT = """You are evaluating whether an interpretability tool's analysis of a language model's internal activation can predict what text the model was about to generate next.

The interpretability tool analyzed the model's activation at the last token of a text passage and produced this description:

<nla_summary>
{body}
</nla_summary>

You are also told the literal final token of the passage (the exact token at which the activation was read; it may be a partial word and may start with a space): "{ftok}"

Based on the description and that final token, which of the following text continuations most likely follows the passage the model was processing?

{choices}

Select the letter of the most appropriate continuation. Think briefly, then respond with your answer in <answer> tags, e.g. <answer>A</answer>."""

FT_ONLY_PROMPT = """A text passage was cut off at some point. You are told ONLY the literal final token of the passage (the exact token it ends with; it may be a partial word and may start with a space): "{ftok}"

Based only on that final token, which of the following text continuations most likely follows the passage?

{choices}

Select the letter of the most appropriate continuation. Think briefly, then respond with your answer in <answer> tags, e.g. <answer>A</answer>."""


def choices_block(options):
    return "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(HERE / "data" / "hard_manifest.json"))
    ap.add_argument("--final-tokens", default=str(HERE / "data" / "final_tokens.json"))
    ap.add_argument("--tok", default=str(HERE / "data" / "qwen_tok"))
    ap.add_argument("--explanations", nargs="+",
                    default=[str(HERE / "results" / "explanations_mat.json"),
                             str(HERE / "results" / "explanations_std.json")])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 120, 256])
    ap.add_argument("--full-rollouts", type=int, default=2)
    ap.add_argument("--budget-rollouts", type=int, default=1)
    ap.add_argument("--reverse-lines", action="store_true",
                    help="reverse each explanation's line order before joining "
                         "(backloading probe; caches stay distinct via prompt text)")
    ap.add_argument("--prefetch", action="store_true",
                    help="fire ALL legs' uncached prompts through one big pool "
                         "first (de-serializes the legs; scoring then replays "
                         "from cache)")
    ap.add_argument("--out", default=str(HERE / "results" / "ft_results.json"))
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    contexts = man["contexts"]
    N = len(contexts)
    keypos = [c["answer_pos"] for c in contexts]
    opts = [tuple(c["options"]) for c in contexts]
    ftok = json.load(open(args.final_tokens))
    tokenizer = AutoTokenizer.from_pretrained(args.tok)

    report = {"meta": {"distractors": "same_document", "with_final_token": True,
                       "budgets": args.budgets,
                       "reverse_lines": args.reverse_lines}}

    def body_of(ent_i, r):
        lines = ent_i["explanations"][r]
        return "\n".join(reversed(lines) if args.reverse_lines else lines)

    def expl_job(i, body):
        prompt = FT_EXPL_PROMPT.format(
            body=body, ftok=ftok[str(contexts[i]["ci"])],
            choices=choices_block(opts[i]))
        return ("ftexpl", prompt)

    def budget_bodies(ent, RB):
        toks = {}
        for i in range(N):
            for r in range(RB):
                toks[(i, r)] = tokenizer.encode(body_of(ent[i], r),
                                                add_special_tokens=False)
        return toks

    # ── optional prefetch: every leg's uncached prompt in ONE pool ────────────
    if args.prefetch:
        seen, jobs = set(), []

        def add(job):
            k = hashlib.sha1((FT_VERSION + "|" + job[0] + "|" + job[1]).encode()).hexdigest()
            if k not in seen and cache.get(k) is None:
                seen.add(k)
                jobs.append(job)

        for i, c in enumerate(contexts):
            add(("fttok", FT_ONLY_PROMPT.format(ftok=ftok[str(c["ci"])],
                                                choices=choices_block(opts[i]))))
        for ef in args.explanations:
            data = json.load(open(ef))
            ent = {e["ci"]: e for e in data["entries"]}
            R = min(args.full_rollouts, data["meta"]["rollouts"])
            RB = min(args.budget_rollouts, data["meta"]["rollouts"])
            for i in range(N):
                for r in range(R):
                    add(expl_job(i, body_of(ent[i], r)))
            toks = budget_bodies(ent, RB)
            for T in args.budgets:
                for i in range(N):
                    for r in range(RB):
                        ids = toks[(i, r)][:T]
                        add(expl_job(i, tokenizer.decode(ids) if ids
                                     else "(no explanation)"))
        print(f"prefetch: {len(jobs)} uncached prompts", flush=True)
        run_jobs(jobs, "prefetch")

    # ── leg 1: token-only baseline ────────────────────────────────────────────
    jobs = [("fttok", FT_ONLY_PROMPT.format(ftok=ftok[str(c["ci"])],
                                            choices=choices_block(opts[i])))
            for i, c in enumerate(contexts)]
    picks = run_jobs(jobs, "tokonly")
    k = sum(int(a == keypos[i]) for i, a in enumerate(picks))
    p, lo, hi = wilson(k, N)
    report["token_only"] = {"acc": p, "ci": [lo, hi], "n": N}
    print(f"token-only baseline: {p:.1%} [{lo:.1%},{hi:.1%}]  (chance 10%)", flush=True)

    # ── legs 2+3 per arm ──────────────────────────────────────────────────────
    report["arms"] = {}
    for ef in args.explanations:
        data = json.load(open(ef))
        tag = data["meta"]["model"]
        ent = {e["ci"]: e for e in data["entries"]}

        # full + ft
        R = min(args.full_rollouts, data["meta"]["rollouts"])
        jobs, idx = [], []
        for i in range(N):
            for r in range(R):
                jobs.append(expl_job(i, body_of(ent[i], r)))
                idx.append(i)
        picks = run_jobs(jobs, f"{tag}:full+ft")
        k = sum(int(a == keypos[i]) for i, a in zip(idx, picks))
        p, lo, hi = wilson(k, len(jobs))
        arm = {"full": {"acc": p, "ci": [lo, hi], "n": len(jobs)}, "curve": {}}
        print(f"{tag} full+ft: {p:.1%} [{lo:.1%},{hi:.1%}]", flush=True)

        # budget + ft (re-tokenized truncation, same as grade_budget)
        toks = {}
        RB = min(args.budget_rollouts, data["meta"]["rollouts"])
        for i in range(N):
            for r in range(RB):
                toks[(i, r)] = tokenizer.encode(body_of(ent[i], r),
                                                add_special_tokens=False)
        for T in args.budgets:
            jobs, idx = [], []
            for i in range(N):
                for r in range(RB):
                    ids = toks[(i, r)][:T]
                    body = tokenizer.decode(ids) if ids else "(no explanation)"
                    jobs.append(expl_job(i, body))
                    idx.append(i)
            picks = run_jobs(jobs, f"{tag}:T{T}+ft")
            k = sum(int(a == keypos[i]) for i, a in zip(idx, picks))
            p, lo, hi = wilson(k, len(jobs))
            arm["curve"][T] = {"acc": p, "ci": [lo, hi], "n": len(jobs)}
            print(f"  {tag} T={T:>3}+ft: {p:.1%} [{lo:.1%},{hi:.1%}]", flush=True)
        report["arms"][tag] = arm

    json.dump(cache, open(CACHE_PATH, "w"))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[write] {args.out}")
    print("FT_DONE")


if __name__ == "__main__":
    main()
