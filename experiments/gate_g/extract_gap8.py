"""Gate G gap-8: extract h_16/h_20/h_24/h_28 at the exact gate_c positions.

Replays gate_c/doc_tokens.jsonl through the base model and reads hidden_states
at the meta2 positions, so rows align 1:1 with acts20.npy / meta2.json. Saves
acts16/24/28; re-extracts acts20 only to verify reproduction (cos vs existing).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

WORK = Path(__file__).parent
GC = WORK.parent / "gate_c"
LAYERS = [16, 20, 24, 28]

meta = json.loads((GC / "meta2.json").read_text())
rows_by_doc = defaultdict(list)
for i, m in enumerate(meta):
    rows_by_doc[m["doc_idx"]].append((i, m["pos"]))
docs = {}
for line in open(GC / "doc_tokens.jsonl"):
    r = json.loads(line)
    docs[r["doc_idx"]] = r["ids"]

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()

N = len(meta)
acts = {L: np.zeros((N, 3584), dtype=np.float32) for L in LAYERS}
done = 0
for doc_idx, rws in rows_by_doc.items():
    ids = docs[doc_idx]
    with torch.inference_mode():
        out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
    for i, pos in rws:
        for L in LAYERS:
            acts[L][i] = out.hidden_states[L][0, pos].float().cpu().numpy()
    done += 1
    if done % 200 == 0:
        print(f"[extract_gap8] {done}/{len(rows_by_doc)} docs", flush=True)

for L in (16, 24, 28):                       # acts20 already exists; don't overwrite
    np.save(GC / f"acts{L}.npy", acts[L])

# verify reproduction against the existing acts20
old20 = np.load(GC / "acts20.npy")
n20 = acts[20] / (np.linalg.norm(acts[20], axis=1, keepdims=True) + 1e-8)
o20 = old20 / (np.linalg.norm(old20, axis=1, keepdims=True) + 1e-8)
cos = float((n20 * o20).sum(1).mean())
print(f"[extract_gap8] saved {LAYERS}; reproduction cos(new h20, old acts20) = {cos:.4f}")
