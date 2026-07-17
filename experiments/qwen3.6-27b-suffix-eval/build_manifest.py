"""Build the Suffix-Prediction eval manifest (LOCAL, CPU-only).

Reimplements the corpus/truncation/MC-item construction from Anthropic's NLA
"Suffix Prediction" eval (transformer-circuits.pub/2026/nla), targeted at the
Qwen3.6-27B matryoshka NLA (and the standard L42 NLA for comparison).

Design decisions (see the review in the experiment README):
  * CORPUS = openbmb/Ultra-FineWeb (en) — the models' TRAINING corpus, so the
    eval is in-distribution. We draw from the TAIL shard (part-2001-of-2048),
    far from where training drew (corpus_slice start=0), for held-out-ness.
  * LAYER = 42 (the 27B extraction layer — NOT 20, which is the 7B pipeline).
  * Truncation position t ~ log-uniform[t_lo, t_hi], capped to leave >=answer_len
    tokens. The plan's [512,1536] window is REJECTED: these models trained on a
    much shorter prefix distribution (median n_raw_tokens ~= 243), so [512,1536]
    would test the sparse long tail, not the operating regime. log-uniform[96,600]
    reproduces the training median while staying above the noisy-early-position
    floor.
  * prefix_ids are stored VERBATIM so GPU-side extraction feeds identical token
    ids (immune to tokenizer-version drift between this box and the GPU box).
  * One fixed option set per context (reused across rollouts AND both models):
    the option set depends only on the true suffix + distractor pool, not on any
    model, so skyline validates items for both arms at once.

Output: data/manifest.json
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SHARD = "datasets/openbmb/Ultra-FineWeb/data/ultrafineweb_en/ultrafineweb-en-part-2001-of-2048.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--tok", default=str(HERE / "data" / "qwen_tok"))
    ap.add_argument("--n-contexts", type=int, default=250)
    ap.add_argument("--rollouts", type=int, default=4)  # recorded; used GPU-side
    ap.add_argument("--min-doc-tokens", type=int, default=640)
    ap.add_argument("--answer-len", type=int, default=32)
    ap.add_argument("--t-lo", type=int, default=96)
    ap.add_argument("--t-hi", type=int, default=600)
    ap.add_argument("--distractor-pool", type=int, default=3200)
    ap.add_argument("--n-options", type=int, default=10)
    ap.add_argument("--row-groups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "manifest.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tok)

    # ── read the tail shard lazily (only the row groups we need) ───────────────
    fs = HfFileSystem(token=open("/home/debian/.hf_token").read().strip())
    pf = pq.ParquetFile(fs.open(args.shard))
    contents, sources = [], []
    for rg in range(min(args.row_groups, pf.num_row_groups)):
        t = pf.read_row_group(rg, columns=["content", "source"])
        contents += t.column("content").to_pylist()
        sources += t.column("source").to_pylist()
    print(f"[read] {len(contents)} raw docs from {args.row_groups} row group(s)")

    # ── tokenize + length filter ───────────────────────────────────────────────
    docs = []
    for c, s in zip(contents, sources):
        if not c or len(c) < 200:
            continue
        ids = tok.encode(c, add_special_tokens=True)
        if len(ids) >= args.min_doc_tokens:
            docs.append({"ids": ids, "text": c, "source": s,
                         "sha": hashlib.sha1(c.encode()).hexdigest()[:16]})
    print(f"[filter] {len(docs)} docs >= {args.min_doc_tokens} tokens")
    rng.shuffle(docs)
    # dedup by content sha (defensive)
    seen, uniq = set(), []
    for d in docs:
        if d["sha"] in seen:
            continue
        seen.add(d["sha"])
        uniq.append(d)
    docs = uniq

    need = args.n_contexts
    eval_docs = docs[:need]
    pool_docs = docs[need:need + args.distractor_pool]
    assert len(eval_docs) == need, f"only {len(eval_docs)} eval docs"
    assert len(pool_docs) >= args.n_options * 2, "distractor pool too small"
    print(f"[split] {len(eval_docs)} eval docs, {len(pool_docs)} distractor-pool docs")

    def loguniform_t(doc_len):
        hi = min(args.t_hi, doc_len - args.answer_len - 1)
        lo = min(args.t_lo, hi)
        u = rng.uniform(math.log(lo), math.log(hi))
        return int(round(math.exp(u)))

    # ── distractor snippet pool: one random 32-tok window per pool doc ─────────
    distractors = []  # {text, src_doc}
    for di, d in enumerate(pool_docs):
        L = len(d["ids"])
        st = rng.randint(0, L - args.answer_len)
        span = d["ids"][st:st + args.answer_len]
        distractors.append({"text": tok.decode(span, skip_special_tokens=True),
                            "src": f"pool{di}"})
    print(f"[distractors] {len(distractors)} snippets")

    # ── build contexts + fixed 10-way option sets ──────────────────────────────
    contexts = []
    for ci, d in enumerate(eval_docs):
        L = len(d["ids"])
        t = loguniform_t(L)
        prefix_ids = d["ids"][:t + 1]           # activation at token t (last)
        answer_ids = d["ids"][t + 1:t + 1 + args.answer_len]
        answer_text = tok.decode(answer_ids, skip_special_tokens=True)
        this_src = f"eval{ci}"
        # 9 distractors, none from this source doc (pool is disjoint from eval,
        # so just sample without replacement); guard against accidental text dupes
        picks, used = [], {answer_text}
        order = list(range(len(distractors)))
        rng.shuffle(order)
        for j in order:
            dt = distractors[j]["text"]
            if dt in used or distractors[j]["src"] == this_src:
                continue
            picks.append(dt)
            used.add(dt)
            if len(picks) == args.n_options - 1:
                break
        assert len(picks) == args.n_options - 1
        options = [answer_text] + picks
        idx = list(range(args.n_options))
        rng.shuffle(idx)
        shuffled = [options[i] for i in idx]
        answer_pos = idx.index(0)               # where the true suffix landed
        contexts.append({
            "ci": ci,
            "source": str(d["source"]),
            "content_sha": d["sha"],
            "doc_len": L,
            "t": t,
            "prefix_ids": prefix_ids,
            "prefix_text": tok.decode(prefix_ids, skip_special_tokens=True),
            "answer_text": answer_text,
            "options": shuffled,
            "answer_pos": answer_pos,
            "answer_letter": "ABCDEFGHIJ"[answer_pos],
        })

    ts = [c["t"] for c in contexts]
    ts_sorted = sorted(ts)
    meta = {
        "shard": args.shard, "layer": 42, "n_contexts": len(contexts),
        "rollouts": args.rollouts, "answer_len": args.answer_len,
        "n_options": args.n_options, "t_lo": args.t_lo, "t_hi": args.t_hi,
        "seed": args.seed, "grader": "anthropic/claude-haiku-4.5",
        "base_model": "Qwen/Qwen3.6-27B",
        "t_dist": {"min": ts_sorted[0], "p50": ts_sorted[len(ts) // 2],
                   "max": ts_sorted[-1]},
    }
    Path(args.out).write_text(json.dumps({"meta": meta, "contexts": contexts}))
    print(f"[write] {args.out}")
    print(f"[meta] {json.dumps(meta, indent=0)}")


if __name__ == "__main__":
    main()
