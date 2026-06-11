"""Rebuild attention evidence LOCALLY from attn_rows.npy + head_norms.npy +
doc_tokens.jsonl, with two structural fixes over extract_d's version:

1. Source context windows are TRUNCATED AT THE POSITION — the original
   ctx = decode(ids[j-3:j+4]) spills past the labeled position when j is
   local, leaking the ground-truth continuation into labels (subagent
   review: ~40/100 labels quoted future tokens). Here ctx ends at pos.
2. Doc-start sources (j == 0, the attention resting state) are dropped
   entirely instead of being narrated-then-dismissed by the labeler.

Writes attn_evidence2.jsonl (same schema as attn_evidence.jsonl, minus
logit-lens fields, which inspection showed are junk for mid-stack writes).
CPU-only; needs the venv (tokenizers).
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

WORK = Path(__file__).parent
TOPK_SRC = 5
MAX_HEADS = 8
HEAD_COVER = 0.90

meta = json.loads((WORK / "meta_d.json").read_text())
rows = np.load(WORK / "attn_rows.npy", mmap_mode="r")     # [N, 28, 512]
norms = np.load(WORK / "head_norms.npy", mmap_mode="r")   # [N, 28]
write_norms = np.linalg.norm(np.load(WORK / "attn_out.npy", mmap_mode="r"),
                             axis=1)  # match extract_d's total_norm semantics
doc_tokens = {}
for line in (WORK / "doc_tokens.jsonl").open():
    r = json.loads(line)
    doc_tokens[r["doc_idx"]] = r["ids"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

with (WORK / "attn_evidence2.jsonl").open("w") as f:
    for i, m in enumerate(meta):
        ids = doc_tokens[m["doc_idx"]]
        pos = m["pos"]
        hn = np.asarray(norms[i], dtype=np.float64)
        fracs = hn / max(hn.sum(), 1e-9)
        order = np.argsort(-fracs)
        heads_ev = []
        cum = 0.0
        for h in order[:MAX_HEADS]:
            if cum >= HEAD_COVER:
                break
            cum += float(fracs[h])
            w = np.asarray(rows[i, h, :pos + 1], dtype=np.float64)
            top = np.argsort(-w)[:TOPK_SRC + 2]
            srcs = []
            for j in top:
                if j == 0 or w[j] < 0.02:      # drop resting state + dust
                    continue
                srcs.append({"j": int(j), "w": round(float(w[j]), 4),
                             "tok": tok.decode([ids[int(j)]]),
                             "ctx": tok.decode(
                                 ids[max(0, int(j) - 3):min(int(j) + 4, pos + 1)])})
                if len(srcs) >= TOPK_SRC:
                    break
            if srcs:
                heads_ev.append({"h": int(h),
                                 "frac": round(float(fracs[h]), 4),
                                 "src": srcs})
        f.write(json.dumps({"i": i,
                            "total_norm": round(float(write_norms[i]), 3),
                            "heads": heads_ev}) + "\n")
        if i % 3000 == 0:
            print(f"[rebuild] {i}/{len(meta)}", flush=True)
print("[rebuild] done -> attn_evidence2.jsonl")
