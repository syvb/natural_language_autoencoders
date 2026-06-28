"""How does steering strength affect WHERE in the AV's list the steered trait first appears?

Genuine L20 direction (the one that verbalizes). For each neutral base, inject
a = base + r*||base||*v_hat renormalized to injection_scale, decode the AV explanation (a
newline-separated list of snippets), and record the list. A separate judge finds the 1-based index
of the first list item mentioning the trait. Sweep r (fine, focused on the low band where the trait
should start LATE and move toward the top as strength rises).

Outputs frontload_v2_raw.json (one row per trait x base x r, with the parsed item list).
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
DIRS = "/workspace/genuine_out/genuine_dirs.npz"
OUT = "/workspace/frontload_out"
LAYER = 20
R_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
TRAITS = ["sycophancy", "neuroticism", "yellow"]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

NEUTRAL_BASES = [
    "The history of papermaking stretches back nearly two thousand years, beginning in China and gradually spreading westward along trade routes over many centuries.",
    "A river system is shaped by the land it crosses, carving valleys, depositing sediment, and slowly changing course as the seasons and the years pass.",
    "In the early morning the market fills with vendors arranging their stalls, stacking crates of produce and calling out the day's prices to passing shoppers.",
    "The library's reading room was quiet except for the turning of pages and the occasional scrape of a chair against the polished wooden floor.",
    "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models to predict conditions several days into the future.",
    "The old clock tower in the town square has marked the hours for generations, its bell ringing out across the rooftops every morning and evening.",
    "Photosynthesis lets plants convert sunlight, water, and carbon dioxide into the sugars they need to grow and the oxygen we breathe.",
    "The train wound slowly through the mountains, past tunnels and bridges, while passengers watched the steep slopes drift by the windows.",
    "Bread requires only a few basic ingredients, yet the way they are combined, kneaded, and baked can produce an endless variety of textures.",
    "The museum's new exhibit traces the development of printing, from carved wooden blocks to movable type to the high-speed presses of today.",
    "Coral reefs support an enormous diversity of marine life, providing shelter and food for countless species within their calcium structures.",
    "The committee met on Thursday to review the budget, discuss upcoming projects, and assign responsibilities for the coming quarter.",
    "A good map balances detail and clarity, showing enough features to be useful without crowding the page so densely it becomes hard to read.",
    "Software is written in layers, with low-level instructions hidden beneath the friendlier interfaces that people actually interact with daily.",
    "The lighthouse stood at the edge of the cliff, its beam sweeping across the dark water to warn ships away from the rocks below.",
    "Limestone caves form slowly as mildly acidic water dissolves the rock over many thousands of years, leaving elaborate formations behind.",
    "The carpenter measured the plank twice before making a single careful cut along the pencilled line near the workbench.",
    "Commuters filled the platform, glancing at the board as the next train approached and shuffling toward the edge with their bags.",
    "The annual report was distributed to shareholders ahead of the spring meeting, summarizing revenue, costs, and plans for expansion.",
    "Surveyors marked the property boundaries with bright stakes and tape before the crew began clearing the overgrown lot.",
]

# ---- extract neutral base activations (training convention) ----
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
del bm
torch.cuda.empty_cache()
V = np.load(DIRS)

# ---- AV ----
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
                      max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


def items_of(expl):
    return [ln.strip() for ln in re.split(r"\n+", expl.strip()) if ln.strip()]


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
        items = items_of(expl)
        rows.append({"trait": trait, "r": r, "base_idx": bi, "n_items": len(items),
                     "items": items, "explanation": expl, "cjk": bool(CJK.search(g))})
    print(f"  av {min(s+B,len(conds))}/{len(conds)}", flush=True)
json.dump(rows, open(f"{OUT}/frontload_v2_raw.json", "w"), indent=1)
print(f"median n_items: {int(np.median([r['n_items'] for r in rows]))} | cjk {sum(r['cjk'] for r in rows)/len(rows):.3f}")
print("FRONTLOAD_V2_DONE", flush=True)
