"""At what steering strength does a v2 probe/CAA direction show up in the AV's explanation?

Take a neutral base activation, add r*||base||*v_hat (the trained direction), renormalize to the
AV's injection_scale (in-distribution norm), inject into the RLed AV, and read the explanation.
Sweep r per trait for both the LR-probe and mean-diff directions, and find the r at which the trait
appears. Positive controls (inject on-trait *text* activations) prove the AV reads traits at all.

  r = 0   -> inject the neutral base (AV describes the neutral text)
  r large -> inject ~ the pure direction (normalized to 150)

Activation extraction follows the training convention (raw text + BOS, layer-20 last token, no
normalization). Directions come from probes_v2.npz. Explanations saved raw -> judged locally.
"""
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer, Qwen2ForCausalLM

from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open

BASE = "/workspace/models/qwen2.5-7b-instruct"
AV = "/workspace/av_ckpt"
PROBES = "/workspace/probes_v2_out/probes_v2.npz"
OUT = "/workspace/verbalize_out"
LAYER = 20
R_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
TRAITS = ["sycophancy", "neuroticism", "yellow"]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

NEUTRAL_BASES = [
    "The history of papermaking stretches back nearly two thousand years, beginning in China and gradually spreading westward along trade routes over many centuries.",
    "A river system is shaped by the land it crosses, carving valleys, depositing sediment, and slowly changing course as the seasons and the years pass.",
    "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models to predict conditions several days into the future.",
    "Photosynthesis allows plants to convert sunlight, water, and carbon dioxide into the sugars they need to grow and the oxygen the rest of us breathe.",
    "The committee met on Thursday to review the budget, discuss upcoming projects, and assign responsibilities for the coming quarter.",
    "Software is written in layers, with low-level instructions hidden beneath the friendlier interfaces that most people actually interact with day to day.",
    "Limestone caves form slowly as mildly acidic water dissolves the rock over thousands of years, leaving behind elaborate formations.",
    "The museum's new exhibit traces the development of printing, from carved wooden blocks to movable type to the high-speed presses of the modern era.",
    "Commuters filled the platform, glancing at the board as the next train approached and shuffling toward the edge with their bags.",
    "A good map balances detail and clarity, showing enough features to be useful without crowding the page so densely that it becomes hard to read.",
]
# positive controls: on-trait raw text whose activation the AV should verbalize
CTRL = {
    "yellow": [
        "Yellow is my favorite color and I surround myself with it: a sunny yellow kitchen, lemon-yellow curtains, and a bright canary bicycle.",
        "She loved everything yellow, from sunflowers and daffodils to her cheerful golden raincoat and the buttery yellow walls of her room.",
    ],
    "neuroticism": [
        "I feel anxious and on edge almost all the time, overwhelmed by worry and dread, unable to stop dwelling on everything that might go wrong.",
        "He was a nervous, fretful person, constantly fearful and insecure, his mood swinging into gloom at the smallest setback.",
    ],
    "sycophancy": [
        "What a brilliant and insightful question! You are absolutely right, your judgment is impeccable, and I completely agree with everything you say.",
        "Of course, you make an excellent point as always; you are so wise and clever that I could not possibly disagree with someone as remarkable as you.",
    ],
}

# ---------------------------------------------------------------- extract activations (base model)
btok = AutoTokenizer.from_pretrained(BASE)
bm = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = bm.model.layers
bm.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))


@torch.no_grad()
def extract(texts):
    out = np.empty((len(texts), 3584), np.float32)
    for k, t in enumerate(texts):
        ids = btok(t, add_special_tokens=True)["input_ids"]
        cap.clear()
        bm.model(input_ids=torch.tensor([ids]).to(dev), use_cache=False)
        out[k] = cap["h"][0, -1].float().cpu().numpy()
    return out


base_acts = extract(NEUTRAL_BASES)
ctrl_acts = {t: extract(CTRL[t]) for t in TRAITS}
del bm
torch.cuda.empty_cache()

# ---------------------------------------------------------------- build conditions
V = np.load(PROBES)
conds = []   # (trait, dir, r, base_idx, raw_vec)
for trait in TRAITS:
    for dname in ["probe", "md"]:
        vh = V[f"{trait}_{dname}_unit"].astype(np.float32)
        for bi, b in enumerate(base_acts):
            bn = float(np.linalg.norm(b))
            for r in R_GRID:
                conds.append((trait, dname, r, bi, (b + r * bn * vh).astype(np.float32)))
ctrl_conds = []
for trait in TRAITS:
    for ci, a in enumerate(ctrl_acts[trait]):
        ctrl_conds.append((trait, "ctrl", -1.0, ci, a.astype(np.float32)))
print(f"conditions: {len(conds)} sweep + {len(ctrl_conds)} controls", flush=True)

# ---------------------------------------------------------------- AV
meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml"))
T = meta["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]
scale = meta["extraction"]["injection_scale"]
actor_tmpl = meta["prompt_templates"]["actor"]
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


# ---------------------------------------------------------------- run
all_conds = conds + ctrl_conds
rows = []
B = 24
for s in range(0, len(all_conds), B):
    chunk = all_conds[s:s + B]
    gens = av_batch([c[4] for c in chunk])
    for (trait, dname, r, idx, vec), g in zip(chunk, gens):
        expl = extract_explanation_open(g) or g
        rows.append({"trait": trait, "dir": dname, "r": r, "idx": idx,
                     "explanation": expl, "cjk": bool(CJK.search(g))})
    print(f"  av {min(s+B,len(all_conds))}/{len(all_conds)}", flush=True)

json.dump(rows, open(f"{OUT}/verbalize_raw.json", "w"), indent=1)
print("VERBALIZE_DONE", flush=True)
