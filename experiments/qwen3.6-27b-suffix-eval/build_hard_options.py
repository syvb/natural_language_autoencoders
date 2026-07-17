"""Specificity test: rebuild the 10-way option sets with SAME-DOCUMENT distractors.

The off-document eval (data/manifest.json) is topic-separable — the true
continuation is the only on-topic option. Here the 9 distractors are OTHER
32-token windows from the SAME document as the true answer, so every option is
on-topic and the grader can no longer win on domain alone: it must identify the
SPECIFIC continuation that follows token t. All 10 windows (true + 9) are
mutually non-overlapping.

Reuses the exact same contexts/activations/explanations; only the option sets
change. Full doc token-ids (not stored in the manifest) are recovered by matching
each context's content_sha back to the tail shard.

Output: data/hard_manifest.json  (same schema as manifest.json)
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SHARD = "datasets/openbmb/Ultra-FineWeb/data/ultrafineweb_en/ultrafineweb-en-part-2001-of-2048.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.json"))
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--tok", default=str(HERE / "data" / "qwen_tok"))
    ap.add_argument("--row-groups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "hard_manifest.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tok)
    man = json.load(open(args.manifest))
    contexts = man["contexts"]
    A = man["meta"]["answer_len"]
    NOPT = man["meta"]["n_options"]
    wanted = {c["content_sha"] for c in contexts}

    # recover full doc text for each needed sha from the shard (sha is cheap; only
    # tokenize the ~250 matched docs)
    fs = HfFileSystem(token=open("/home/debian/.hf_token").read().strip())
    pf = pq.ParquetFile(fs.open(args.shard))
    sha2content = {}
    for rg in range(min(args.row_groups, pf.num_row_groups)):
        for c in pf.read_row_group(rg, columns=["content"]).column("content").to_pylist():
            if not c:
                continue
            s = hashlib.sha1(c.encode()).hexdigest()[:16]
            if s in wanted and s not in sha2content:
                sha2content[s] = c
    missing = wanted - set(sha2content)
    assert not missing, f"{len(missing)} content_sha not found in shard"
    sha2ids = {s: tok.encode(c, add_special_tokens=True) for s, c in sha2content.items()}
    print(f"[recover] {len(sha2ids)} docs re-tokenized from shard")

    def nonoverlapping_windows(ids, t, need):
        """Greedily pick `need` 32-token windows, mutually non-overlapping and not
        overlapping the true window [t+1, t+33)."""
        L = len(ids)
        reserved = [(t + 1, t + 1 + A)]                     # true window

        def ok(s):
            e = s + A
            return all(e <= a or s >= b for (a, b) in reserved)

        starts = list(range(0, L - A + 1))
        rng.shuffle(starts)
        picks = []
        for s in starts:
            if len(picks) == need:
                break
            if ok(s):
                picks.append(s)
                reserved.append((s, s + A))
        return picks

    out_contexts, short = [], 0
    for c in contexts:
        ids = sha2ids[c["content_sha"]]
        t = c["t"]
        starts = nonoverlapping_windows(ids, t, NOPT - 1)
        # decode, dedup identical text and vs the true answer
        answer = c["answer_text"]
        picks, used = [], {answer}
        for s in starts:
            dt = tok.decode(ids[s:s + A], skip_special_tokens=True)
            if dt in used:
                continue
            picks.append(dt)
            used.add(dt)
        if len(picks) < NOPT - 1:                            # extremely rare
            short += 1
            continue
        picks = picks[: NOPT - 1]
        options = [answer] + picks
        idx = list(range(NOPT))
        rng.shuffle(idx)
        shuffled = [options[i] for i in idx]
        answer_pos = idx.index(0)
        out_contexts.append({**c, "options": shuffled, "answer_pos": answer_pos,
                             "answer_letter": "ABCDEFGHIJ"[answer_pos]})

    meta = {**man["meta"], "distractor_kind": "same_document",
            "n_contexts": len(out_contexts)}
    Path(args.out).write_text(json.dumps({"meta": meta, "contexts": out_contexts}))
    print(f"[write] {args.out}: {len(out_contexts)} contexts "
          f"({short} dropped for too few same-doc windows)")


if __name__ == "__main__":
    main()
