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
    for attempt in range(3):
        if attempt == 2:
            body_json["messages"][0]["content"] = prompt + \
                "\n\nRespond with exactly one letter in <answer> tags."
        try:
            req = urllib.request.Request(
                URL, json.dumps(body_json).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            m = ANS_RE.search(r["choices"][0]["message"]["content"])
            if m:
                return k, LETTERS.index(m.group(1).upper())
        except Exception:
            continue
    return k, None


def run_jobs(jobs, tag):
    with ThreadPoolExecutor(max_workers=64) as ex:
        res = list(ex.map(ask_raw, jobs))
    fails = 0
    for kk, v in res:
        cache[kk] = v
        if v is None:
            fails += 1
    json.dump(cache, open(CACHE_PATH, "w"))
    print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
    return [v for (_, v) in res]

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
                       "budgets": args.budgets}}

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

        def expl_job(i, body):
            prompt = FT_EXPL_PROMPT.format(
                body=body, ftok=ftok[str(contexts[i]["ci"])],
                choices=choices_block(opts[i]))
            return ("ftexpl", prompt)

        # full + ft
        R = min(args.full_rollouts, data["meta"]["rollouts"])
        jobs, idx = [], []
        for i in range(N):
            for r in range(R):
                jobs.append(expl_job(i, "\n".join(ent[i]["explanations"][r])))
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
                body = "\n".join(ent[i]["explanations"][r])
                toks[(i, r)] = tokenizer.encode(body, add_special_tokens=False)
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
