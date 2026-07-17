"""Budget axis: suffix-prediction accuracy vs. how many explanation TOKENS the
grader is allowed to read — the matryoshka front-loading test.

TOKEN truncation (not lines): the matryoshka is RL-trained on random-length
truncation of U[1,120] *content tokens* of the explanation, so tokens are the
unit the matryoshka property is defined in, and "first N tokens" is comparable
across the two arms (which use different line granularities). We tokenize each
explanation with the base model's tokenizer, truncate to the first N tokens,
detokenize, and grade against the SAME fixed option set. Shares the grader cache
with grade_suffix.py.

Run with the CPU venv python (needs transformers for the tokenizer):
    /home/debian/nanoNLA-multi-input/.venv-cpu/bin/python grade_budget.py \
        --explanations results/explanations_mat.json results/explanations_std.json
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from grade_suffix import ask, run_jobs, wilson, cache, CACHE_PATH  # reuse infra

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.json"))
    ap.add_argument("--tok", default=str(HERE / "data" / "qwen_tok"))
    ap.add_argument("--explanations", nargs="+", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64, 120])
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results" / "budget.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tok)
    man = json.load(open(args.manifest))
    contexts = man["contexts"]
    N = len(contexts)
    keypos = [c["answer_pos"] for c in contexts]
    opts = [tuple(c["options"]) for c in contexts]

    report = {"budgets": args.budgets, "rollouts": args.rollouts,
              "unit": "content_tokens", "arms": {}}
    for ef in args.explanations:
        data = json.load(open(ef))
        tag = data["meta"]["model"]
        ent = {e["ci"]: e for e in data["entries"]}
        R = min(args.rollouts, data["meta"]["rollouts"])

        # pre-tokenize each (ctx, rollout) explanation once
        toks = {}      # (i, r) -> token-id list of the full explanation
        full_len = []  # token length of each explanation (for reporting)
        for i in range(N):
            for r in range(R):
                body = "\n".join(ent[i]["explanations"][r])
                ids = tok.encode(body, add_special_tokens=False)
                toks[(i, r)] = ids
                full_len.append(len(ids))
        fl = sorted(full_len)
        med_len = fl[len(fl) // 2] if fl else 0

        curve = {}
        for T in args.budgets:
            jobs, idx = [], []
            for i in range(N):
                for r in range(R):
                    ids = toks[(i, r)][:T]
                    body = tok.decode(ids) if ids else "(no explanation)"
                    jobs.append(("expl", body, opts[i]))
                    idx.append(i)
            picks = run_jobs(jobs, f"{tag}:T{T}")
            kk = sum(int(a == keypos[i]) for i, a in zip(idx, picks))
            nn = len(picks)
            p, lo, hi = wilson(kk, nn)
            curve[T] = {"acc": p, "ci": [lo, hi], "n": nn}
            print(f"  {tag} T={T:>3}: {p:.1%} [{lo:.1%},{hi:.1%}] (n={nn})", flush=True)
        report["arms"][tag] = {"curve": curve, "expl_token_len_median": med_len}
        print(f"  [{tag}] explanation token length: median {med_len}", flush=True)

    json.dump(cache, open(CACHE_PATH, "w"))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[write] {args.out}")
    print("BUDGET_DONE")


if __name__ == "__main__":
    main()
