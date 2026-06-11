"""armT' (improvement cycle): enriched templated labels from FULL patterns.

Differences vs armT (built from the sparse top-8-head evidence):
- Source importance = sum over ALL 28 heads of (head contribution fraction
  x attention weight) — the contribution-weighted pattern, not top-5-of-8.
- Distant sources (>15 tokens back) listed first with context, local
  carry-over compressed to one clause, dedup across heads.
- Write strength stated from the norm percentile (weak/moderate/strong).
- Context windows truncated at the position (no future tokens).

Reuses the exact train/eval position sets of armT_8k for comparability.
Outputs armT2_{train8k,eval8k}.json.
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

WORK = Path(__file__).parent
N_DISTANT = 8
LOCAL_WINDOW = 15
W_FLOOR = 0.005

meta = json.loads((WORK / "meta_d.json").read_text())
rows = np.load(WORK / "attn_rows.npy", mmap_mode="r")
hnorms = np.load(WORK / "head_norms.npy", mmap_mode="r")
write_norms = np.linalg.norm(np.load(WORK / "attn_out.npy", mmap_mode="r"), axis=1)
q_lo, q_hi = np.percentile(write_norms, [10, 75])
doc_tokens = {}
for line in (WORK / "doc_tokens.jsonl").open():
    r = json.loads(line)
    doc_tokens[r["doc_idx"]] = r["ids"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

train_tidx = [r["tidx"] for r in json.loads((WORK / "armT_train8k.json").read_text())]
eval_tidx = [r["tidx"] for r in json.loads((WORK / "armT_eval8k.json").read_text())]


def label_for(i):
    m = meta[i]
    ids = doc_tokens[m["doc_idx"]]
    pos = m["pos"]
    hn = np.asarray(hnorms[i], dtype=np.float64)
    fr = hn / max(hn.sum(), 1e-9)
    # contribution-weighted source importance over all heads
    imp = (fr[:, None] * np.asarray(rows[i, :, :pos + 1], dtype=np.float64)).sum(0)
    imp[0] = 0.0  # resting state
    order = np.argsort(-imp)
    distant, local = [], []
    seen = set()
    for j in order:
        if imp[j] < W_FLOOR:
            break
        t = tok.decode([ids[j]]).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        if pos - j > LOCAL_WINDOW:
            if len(distant) < N_DISTANT:
                ctx = tok.decode(ids[max(0, j - 4):min(j + 5, pos + 1)]).strip()
                distant.append((t, ctx, imp[j]))
        else:
            local.append(t)
    wn = write_norms[i]
    strength = ("a weak" if wn < q_lo else
                "a strong" if wn > q_hi else "a moderate")
    tk = m["token_at_pos"].strip()
    parts = []
    if distant:
        big = [f"\"{t}\" (from \"{c}\")" for t, c, v in distant[:4]]
        small = [f"\"{t}\"" for t, c, v in distant[4:]]
        s = f"pulls in earlier content: {', '.join(big)}"
        if small:
            s += f"; more faintly {', '.join(small)}"
        parts.append(s)
    if local:
        parts.append(f"carries over the local phrase around "
                     f"{', '.join(repr(t) for t in local[:4])}")
    if not parts:
        parts.append("retrieves nothing beyond the position itself")
    return (f"At '{tk}', {strength} attention write " + "; and ".join(parts) + ".")


for name, tidxs in [("armT2_train8k", train_tidx), ("armT2_eval8k", eval_tidx)]:
    out = [{"text": label_for(i), "tidx": i} for i in tidxs]
    (WORK / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: {len(out)}")
print("--- samples:")
for i in [train_tidx[0], train_tidx[5], eval_tidx[0]]:
    print(f"[{i}]", label_for(i)[:400], "\n")
