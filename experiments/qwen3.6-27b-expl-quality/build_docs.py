"""Build the explanation-quality eval manifest (LOCAL, CPU-only).

Two held-out domains, ONE activation position each, for a broad LLM-judged
quality eval (usefulness / hallucination / coherence) of the matryoshka vs
standard 27B NLA explanations:

  * pretrain (250): Ultra-FineWeb-en documents from SHARD part-2002-of-2048 —
    disjoint from the suffix eval's part-2001 and from the training slice
    (early shards, corpus_slice start=0). Protocol otherwise matches
    ../qwen3.6-27b-suffix-eval/build_manifest.py.
  * wildchat (250): conversations sampled from the evalsuite's
    control_convos.json (1,000 English non-toxic WildChat convos), rendered
    with the base Qwen chat template — the same protocol as the eval-awareness
    controls (control_gen.py). Chat data is entirely outside NLA training.

Position t ~ log-uniform[96, 600] (training operating regime) in BOTH domains,
capped so at least 100 continuation tokens remain. A 100-token TRUE
CONTINUATION is stored per context so the judge can fact-check the
explanation's predictive claims. prefix_ids stored VERBATIM so GPU-side
extraction is immune to tokenizer drift. WildChat prefix/continuation text is
decoded WITH special tokens so the judge sees the transcript structure.

Output: data/manifest_quality.json
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
SHARD = "datasets/openbmb/Ultra-FineWeb/data/ultrafineweb_en/ultrafineweb-en-part-2002-of-2048.parquet"
CONVOS = HERE.parent / "qwen3.6-27b-evalsuite" / "control_convos.json"
CONT_LEN = 100


def sample_t(rng, lo, hi):
    u = rng.uniform(math.log(lo), math.log(hi))
    return int(round(math.exp(u)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--convos", default=str(CONVOS))
    ap.add_argument("--tok", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--n-pretrain", type=int, default=250)
    ap.add_argument("--n-wildchat", type=int, default=250)
    ap.add_argument("--min-doc-tokens", type=int, default=640)
    ap.add_argument("--min-chat-tokens", type=int, default=240)
    ap.add_argument("--max-chat-tokens", type=int, default=2048)
    ap.add_argument("--t-lo", type=int, default=96)
    ap.add_argument("--t-hi", type=int, default=600)
    ap.add_argument("--max-row-groups", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "manifest_quality.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tok)
    contexts = []

    def add_context(domain, ids, source, sha, skip_special):
        L = len(ids)
        hi = min(args.t_hi, L - CONT_LEN - 1)
        lo = min(args.t_lo, hi)
        t = sample_t(rng, lo, hi)
        prefix_ids = ids[: t + 1]               # activation at token t (last)
        cont_ids = ids[t + 1: t + 1 + CONT_LEN]
        contexts.append({
            "ci": len(contexts), "domain": domain,
            "source": source, "content_sha": sha, "doc_len": L, "t": t,
            "prefix_ids": prefix_ids,
            "prefix_text": tok.decode(prefix_ids, skip_special_tokens=skip_special),
            "continuation_text": tok.decode(cont_ids, skip_special_tokens=skip_special),
        })

    # ── domain 1: pretraining docs ────────────────────────────────────────────
    fs = HfFileSystem(token=open("/home/debian/.hf_token").read().strip())
    pf = pq.ParquetFile(fs.open(args.shard))
    docs, seen = [], set()
    for rg in range(min(args.max_row_groups, pf.num_row_groups)):
        t = pf.read_row_group(rg, columns=["content", "source"])
        for c, s in zip(t.column("content").to_pylist(),
                        t.column("source").to_pylist()):
            if not c or len(c) < 200:
                continue
            sha = hashlib.sha1(c.encode()).hexdigest()[:16]
            if sha in seen:
                continue
            seen.add(sha)
            ids = tok.encode(c, add_special_tokens=True)
            if len(ids) >= args.min_doc_tokens:
                docs.append({"ids": ids, "source": str(s), "sha": sha})
        print(f"[read] row group {rg}: {len(docs)} usable docs so far", flush=True)
        if len(docs) >= args.n_pretrain * 2:
            break
    assert len(docs) >= args.n_pretrain, f"only {len(docs)} docs"
    rng.shuffle(docs)
    for d in docs[: args.n_pretrain]:
        add_context("pretrain", d["ids"], d["source"], d["sha"], True)

    # ── domain 2: WildChat conversations ──────────────────────────────────────
    convos = json.load(open(args.convos))
    order = list(range(len(convos)))
    rng.shuffle(order)
    n_wc = 0
    for wi in order:
        if n_wc == args.n_wildchat:
            break
        msgs = convos[wi]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            continue
        ids = ids[: args.max_chat_tokens]
        if len(ids) < args.min_chat_tokens:
            continue
        sha = hashlib.sha1(text.encode()).hexdigest()[:16]
        add_context("wildchat", ids, f"wildchat:{wi}", sha, False)
        n_wc += 1
    assert n_wc == args.n_wildchat, f"only {n_wc} usable convos"

    ts = sorted(c["t"] for c in contexts)
    meta = {"shard": args.shard, "convos": str(args.convos), "layer": 42,
            "n_contexts": len(contexts),
            "n_pretrain": args.n_pretrain, "n_wildchat": args.n_wildchat,
            "cont_len": CONT_LEN, "t_lo": args.t_lo, "t_hi": args.t_hi,
            "seed": args.seed, "base_model": "Qwen/Qwen3.6-27B",
            "t_dist": {"min": ts[0], "p50": ts[len(ts) // 2], "max": ts[-1]}}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "contexts": contexts}))
    print(f"[write] {args.out}")
    print(f"[meta] {json.dumps(meta)}")


if __name__ == "__main__":
    main()
