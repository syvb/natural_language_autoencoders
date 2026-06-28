"""AV verbalization sweep with the GENUINE L20 direction (raw-text neutral-negative).

Mirror of verbalize_steer.py but using the genuine direction instead of the A/B-answer one:
a = base + r*||base||*v_hat_genuine_L20, renormalized to injection_scale, injected into the AV.
Sweep r and judge whether the trait now shows up. Output -> judge with judge_v2.py.
"""
import json
import os
import re

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, Qwen2ForCausalLM

from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open

AV = "/workspace/av_ckpt"
DIRS = "/workspace/genuine_out/genuine_dirs.npz"
OUT = "/workspace/genuine_out"
R_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
TRAITS = ["sycophancy", "neuroticism", "yellow"]
dev = "cuda"

V = np.load(DIRS)
base_acts = V["av_base_acts"]   # neutral L20 base activations
m = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = m["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]
scale = m["extraction"]["injection_scale"]
actor_tmpl = m["prompt_templates"]["actor"]
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
ptext = atok.apply_chat_template([{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}],
                                 add_generation_prompt=True, tokenize=False)
prompt_ids = atok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
S = prompt_ids.shape[1]
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs):
    n = len(vecs)
    ids = prompt_ids.repeat(n, 1)
    e = emb(ids)
    Vt = normalize_activation(torch.from_numpy(np.stack(vecs)).to(dev), scale)
    e2 = inject_at_marked_positions(ids, e, Vt, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, S, device=dev),
                      max_new_tokens=200, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


conds = []
for trait in TRAITS:
    vh = V[f"{trait}_L20_genuine_unit"].astype(np.float32)
    for bi, b in enumerate(base_acts):
        bn = float(np.linalg.norm(b))
        for r in R_GRID:
            conds.append((trait, r, bi, (b + r * bn * vh).astype(np.float32)))
print(f"conditions: {len(conds)}", flush=True)

rows = []
B = 24
for s in range(0, len(conds), B):
    chunk = conds[s:s + B]
    for (trait, r, bi, vec), g in zip(chunk, av_batch([c[3] for c in chunk])):
        expl = extract_explanation_open(g) or g
        rows.append({"trait": trait, "r": r, "base_idx": bi, "explanation": expl, "cjk": bool(CJK.search(g))})
    print(f"  av {min(s+B,len(conds))}/{len(conds)}", flush=True)
json.dump(rows, open(f"{OUT}/genuine_verbalize_raw.json", "w"), indent=1)
print(f"cjk rate {sum(r['cjk'] for r in rows)/len(rows):.3f}. GENUINE_VERBALIZE_DONE", flush=True)
