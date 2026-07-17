"""Grade the Suffix-Prediction eval (LOCAL, OpenRouter/Haiku 4.5).

Blind grader sees ONLY the AV explanation (never the source passage) and the ten
options, and picks the true 32-token continuation. Chance = 10%.

Controls:
  * shuffled-pairing: re-pair each explanation with a DERANGED context's option
    set (donor j = (i+1)%N) and score against donor j's key. This control exists
    to catch an ABOVE-chance result, which would mean the grader exploits a
    spurious cue (option length/format) rather than content. At-or-below chance
    (~10%) passes: below-chance just means a topic-A explanation actively steers
    away from topic-B's true answer — no positive cue exists. NB it rules out
    format/length cues, NOT topic separability (distractors are off-document).
  * skyline: replace the explanation with the actual prefix passage. Expect near
    100%; validates that items are solvable and distractors aren't plausible.
    Model-independent ⇒ run once, shared across arms.
  * outlier split: primary accuracy is also reported dropping the high-norm
    (>3x median) activations flagged GPU-side.

Prompt is VERBATIM from the paper's eval widget. Never pass the passage to the
primary/shuffled grader.

Usage:  python grade_suffix.py --explanations explanations_mat.json explanations_std.json
        (writes results/results.json and prints the summary)
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
GRADER = "anthropic/claude-haiku-4.5"
CACHE_PATH = HERE / "results" / ".suffix_grade_cache.json"
PROMPT_VERSION = "nla-suffix-v1"
LETTERS = "ABCDEFGHIJ"
ANS_RE = re.compile(r"<answer>\s*([A-J])\s*</answer>", re.I)

EXPL_PROMPT = """You are evaluating whether an interpretability tool's analysis of a language model's internal activation can predict what text the model was about to generate next.

The interpretability tool analyzed the model's activation at the last token of a text passage and produced this description:

<nla_summary>
{body}
</nla_summary>

Based on this description, which of the following text continuations most likely follows the passage the model was processing?

{choices}

Select the letter of the most appropriate continuation. Think briefly, then respond with your answer in <answer> tags, e.g. <answer>A</answer>."""

# skyline: minimally-adjusted framing — the passage itself in place of the summary
SKY_PROMPT = """You are evaluating whether the text a language model was processing lets you predict what it was about to generate next.

Here is the passage the model was processing, up to the last token:

<passage>
{body}
</passage>

Based on this passage, which of the following text continuations most likely follows it?

{choices}

Select the letter of the most appropriate continuation. Think briefly, then respond with your answer in <answer> tags, e.g. <answer>A</answer>."""

cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}


def choices_block(options):
    return "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(options))


def ask(job):
    """job = (kind, body, options_tuple). Returns (cache_key, letter_index|None)."""
    kind, body, options = job
    tmpl = SKY_PROMPT if kind == "sky" else EXPL_PROMPT
    prompt = tmpl.format(body=body, choices=choices_block(options))
    k = hashlib.sha1((PROMPT_VERSION + "|" + kind + "|" + prompt).encode()).hexdigest()
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
            txt = r["choices"][0]["message"]["content"]
            m = ANS_RE.search(txt)
            if m:
                return k, LETTERS.index(m.group(1).upper())
        except Exception:
            continue
    return k, None   # unparseable ⇒ scored incorrect downstream


def run_jobs(jobs, tag):
    with ThreadPoolExecutor(max_workers=32) as ex:
        res = list(ex.map(ask, jobs))
    fails = 0
    for (k, v) in res:
        cache[k] = v
        if v is None:
            fails += 1
    json.dump(cache, open(CACHE_PATH, "w"))
    print(f"[{tag}] {len(jobs)} calls, {fails} unparsed", flush=True)
    return [v for (_, v) in res]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - hw) / d, (c + hw) / d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.json"))
    ap.add_argument("--explanations", nargs="+", required=True)
    ap.add_argument("--out", default=str(HERE / "results" / "results.json"))
    args = ap.parse_args()
    os.makedirs(HERE / "results", exist_ok=True)

    man = json.load(open(args.manifest))
    contexts = man["contexts"]
    N = len(contexts)
    keypos = [c["answer_pos"] for c in contexts]
    opts = [tuple(c["options"]) for c in contexts]

    report = {"meta": man["meta"], "arms": {}}

    # ── skyline (once) ────────────────────────────────────────────────────────
    sky_jobs = [("sky", contexts[i]["prefix_text"], opts[i]) for i in range(N)]
    sky = run_jobs(sky_jobs, "skyline")
    sky_correct = sum(int(a == keypos[i]) for i, a in enumerate(sky) if a is not None)
    p, lo, hi = wilson(sky_correct, N)
    report["skyline"] = {"acc": p, "ci": [lo, hi], "n": N, "correct": sky_correct}
    print(f"  skyline acc = {p:.1%}  [{lo:.1%},{hi:.1%}]  (want >90%)", flush=True)

    # ── per-arm primary + shuffled ────────────────────────────────────────────
    for ef in args.explanations:
        data = json.load(open(ef))
        tag = data["meta"]["model"]
        ent = {e["ci"]: e for e in data["entries"]}
        R = data["meta"]["rollouts"]

        # explanations are saved as per-line lists; join to the text the model
        # actually emitted (NOT the Python list repr) before grading.
        def body_of(i, r):
            return "\n".join(ent[i]["explanations"][r])

        # primary: (ci, r) -> grade expl against own options
        prim_jobs, prim_idx = [], []
        for i in range(N):
            for r in range(R):
                prim_jobs.append(("expl", body_of(i, r), opts[i]))
                prim_idx.append((i, r))
        prim = run_jobs(prim_jobs, f"{tag}:primary")

        # shuffled: expl_i vs donor j=(i+1)%N options, score vs donor key
        shuf_jobs, shuf_idx = [], []
        for i in range(N):
            j = (i + 1) % N
            for r in range(R):
                shuf_jobs.append(("expl", body_of(i, r), opts[j]))
                shuf_idx.append((i, j, r))
        shuf = run_jobs(shuf_jobs, f"{tag}:shuffled")

        # metrics — unparseable grade (a is None) counts as INCORRECT (in denom)
        outlier = {e["ci"]: e["norm_outlier"] for e in data["entries"]}
        n_all = len(prim)
        k_all = sum(int(a == keypos[i]) for (i, r), a in zip(prim_idx, prim))
        p_all, lo_all, hi_all = wilson(k_all, n_all)

        # dropping high-norm-outlier activations
        k_in = sum(int(a == keypos[i]) for (i, r), a in zip(prim_idx, prim) if not outlier[i])
        n_in = sum(1 for (i, r) in prim_idx if not outlier[i])
        p_in, lo_in, hi_in = wilson(k_in, n_in)

        # per-context majority vote
        by_ctx = {i: [] for i in range(N)}
        for (i, r), a in zip(prim_idx, prim):
            by_ctx[i].append(a)
        mv_correct, agree = 0, []
        for i in range(N):
            picks = [a for a in by_ctx[i] if a is not None]
            if not picks:
                continue
            maj = max(set(picks), key=picks.count)
            mv_correct += int(maj == keypos[i])
            agree.append(picks.count(maj) / len(picks))
        p_mv, lo_mv, hi_mv = wilson(mv_correct, N)

        # shuffled accuracy (vs donor key) — None counts as a non-match
        sh_k = sum(int(a == keypos[j]) for (i, j, r), a in zip(shuf_idx, shuf))
        sh_n = len(shuf)
        p_sh, lo_sh, hi_sh = wilson(sh_k, sh_n)

        ent_out = {
            "primary": {"acc": p_all, "ci": [lo_all, hi_all], "n": n_all, "correct": k_all},
            "primary_no_outliers": {"acc": p_in, "ci": [lo_in, hi_in], "n": n_in},
            "majority_vote": {"acc": p_mv, "ci": [lo_mv, hi_mv], "n": N},
            "inter_rollout_agreement": sum(agree) / len(agree) if agree else 0,
            "shuffled": {"acc": p_sh, "ci": [lo_sh, hi_sh], "n": sh_n},
            "cjk_explanations": data["meta"].get("cjk_explanations"),
            "n_outliers": data["meta"].get("n_outliers"),
        }
        report["arms"][tag] = ent_out
        print(f"\n=== {tag} ({data['meta']['arm']}) ===", flush=True)
        print(f"  primary        {p_all:.1%}  [{lo_all:.1%},{hi_all:.1%}]  (n={n_all}, chance 10%)")
        print(f"  no-outliers    {p_in:.1%}  [{lo_in:.1%},{hi_in:.1%}]  (n={n_in})")
        print(f"  majority-vote  {p_mv:.1%}  [{lo_mv:.1%},{hi_mv:.1%}]  (n={N})")
        print(f"  agreement      {ent_out['inter_rollout_agreement']:.1%}")
        print(f"  shuffled ctrl  {p_sh:.1%}  [{lo_sh:.1%},{hi_sh:.1%}]  (want ~10%)")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[write] {args.out}")
    print("SUFFIX_GRADE_DONE")


if __name__ == "__main__":
    main()
