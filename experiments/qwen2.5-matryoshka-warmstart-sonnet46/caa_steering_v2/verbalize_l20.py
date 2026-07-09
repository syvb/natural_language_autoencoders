"""Phase C, part 2: inject the propagated L20 acts into the AV and read the explanation.

Loads steered_l20.npz (L20 activations produced by early-layer steering of neutral text), injects
each into the AV (normalized to injection_scale), and saves the explanation. Steered rows ->
verbalize_l20_raw.json (judged downstream); baseline rows -> a txt for eyeballing.
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
IN = "/workspace/steer_l20_out"
OUT = "/workspace/steer_l20_out"
dev = "cuda"

acts = np.load(f"{IN}/steered_l20.npz")["acts"]
meta = json.load(open(f"{IN}/steered_l20_meta.json"))

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


gens = [None] * len(acts)
B = 24
for s in range(0, len(acts), B):
    for k, g in enumerate(av_batch([acts[s + k] for k in range(min(B, len(acts) - s))])):
        gens[s + k] = g
    print(f"  av {min(s+B,len(acts))}/{len(acts)}", flush=True)

steered_rows, baseline_lines = [], []
for md, g in zip(meta, gens):
    expl = extract_explanation_open(g) or g
    if md["trait"] == "baseline":
        baseline_lines.append(f"[base#{md['base_idx']}] {expl[:200]}")
    else:
        steered_rows.append({"trait": md["trait"], "layer": md["layer"], "c": md["c"],
                             "base_idx": md["base_idx"], "explanation": expl, "cjk": bool(CJK.search(g))})
json.dump(steered_rows, open(f"{OUT}/verbalize_l20_raw.json", "w"), indent=1)
open(f"{OUT}/verbalize_l20_baseline.txt", "w").write("\n".join(baseline_lines))
print(f"steered rows: {len(steered_rows)} | cjk rate: {sum(r['cjk'] for r in steered_rows)/max(1,len(steered_rows)):.3f}")
print("VERBALIZE_L20_DONE", flush=True)
