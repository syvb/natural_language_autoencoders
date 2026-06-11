"""Dump FULL attention rows + all-head contribution norms at the sampled
positions (the sparse evidence in attn_evidence.jsonl keeps only dominant
heads / top-5 sources; this keeps everything for later per-head analysis).

Outputs:
  attn_rows.npy  [N, n_heads, MAX_TOKENS] fp16 — attention from position p
                 over all source positions (zero-padded past doc length)
  head_norms.npy [N, n_heads] fp32 — per-head write contribution norms at p

Deterministic re-derivation of extract_d positions via meta_d.json +
doc_tokens.jsonl (no corpus re-stream, no RNG).
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
LAYER = 20
MAX_TOKENS = 512

meta = json.loads((WORK / "meta_d.json").read_text())
doc_tokens = {}
for line in (WORK / "doc_tokens.jsonl").open():
    r = json.loads(line)
    doc_tokens[r["doc_idx"]] = r["ids"]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
    attn_implementation="eager", device_map="cuda").eval()
torch.set_grad_enabled(False)
layer = model.model.layers[LAYER]
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
W_O = layer.self_attn.o_proj.weight.float()

cap = {}
layer.self_attn.o_proj.register_forward_pre_hook(
    lambda m, a: cap.__setitem__("oin", a[0].detach()))

by_doc = {}
for k, m in enumerate(meta):
    by_doc.setdefault(m["doc_idx"], []).append((k, m["pos"]))

rows = np.zeros((len(meta), n_heads, MAX_TOKENS), dtype=np.float16)
norms = np.zeros((len(meta), n_heads), dtype=np.float32)
for nd, (doc_idx, items) in enumerate(sorted(by_doc.items())):
    ids = doc_tokens[doc_idx]
    x = torch.tensor([ids], device="cuda")
    out = model(x, output_attentions=True)
    attw = out.attentions[LAYER][0].float()  # [n_heads, seq, seq]
    oin_seq = cap["oin"][0]
    for k, pos in items:
        rows[k, :, :len(ids)] = attw[:, pos, :].cpu().numpy().astype(np.float16)
        oin = oin_seq[pos].float().view(n_heads, head_dim)
        w = torch.stack([W_O[:, h * head_dim:(h + 1) * head_dim] @ oin[h]
                         for h in range(n_heads)])
        norms[k] = w.norm(dim=-1).cpu().numpy()
    if nd % 200 == 0:
        print(f"[attn_rows] {nd}/{len(by_doc)} docs", flush=True)

np.save(WORK / "attn_rows.npy", rows)
np.save(WORK / "head_norms.npy", norms)
print(f"[attn_rows] saved {rows.shape}, {norms.shape}")
