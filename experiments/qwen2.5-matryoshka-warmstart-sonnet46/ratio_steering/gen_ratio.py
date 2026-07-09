"""Two-concept ratio test: does the ORDER of two steered concepts track their injected ratio?

Inject a = b + ||b|| * (r_y * v_yellow + r_s * v_syco) into 40 neutral base activations,
holding the total steering strength R = sqrt(r_y^2 + r_s^2) fixed while sweeping the
log-ratio lam = log2(r_y / r_s). The matryoshka NLA is trained to order its list by
reconstruction-salience, so which concept it mentions FIRST should track lam; an NLA with
no ordering incentive should be much flatter. Judged downstream by judge_ratio.py.

Same conventions as caa_steering_v2/frontload_v2.py: genuine L20 contrast directions,
prompt template from the AV's own sidecar, injection-time renormalization, T=1 sampling.

Outputs $OUT/$OUT_NAME: one row per (R, lam, rep, base).
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
AV = os.environ.get("AV", "/workspace/av_ckpt")
DIRS = "/workspace/genuine_out/genuine_dirs.npz"
OUT = os.environ.get("OUT", "/workspace/ratio_out")
OUT_NAME = os.environ.get("OUT_NAME", "ratio_raw.json")
LAYER = 20
R_LIST = [0.6, 0.9, 1.3]                                   # total steering strength
LAM_GRID = [-3.0, -2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0, 3.0]  # log2(r_yellow / r_syco)
REPS = 2
_T = float(os.environ.get("NLA_GEN_TEMP", "1"))
GEN_KW = (dict(do_sample=True, temperature=_T, top_p=1.0, top_k=0)
          if _T > 0 else dict(do_sample=False))
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
    "The accountant reconciled the figures line by line before closing the ledger for the month.",
    "A delivery van idled at the curb while the driver checked the address list on a clipboard.",
    "The lecture hall slowly emptied as students filed out into the late afternoon air.",
    "Wind turbines turned steadily on the ridge above the quiet farming valley below.",
    "The archivist scanned the fragile letters before returning them carefully to cold storage.",
    "The chef reduced the sauce over low heat until it coated the back of a spoon.",
    "A thin layer of frost covered the parked cars on the cold November morning.",
    "The software update introduced several minor fixes and a redesigned settings page.",
    "Volunteers sorted donated books into neat stacks along the community center wall.",
    "The tide pulled back to reveal smooth stones and tangled ribbons of kelp on the sand.",
    "The janitor swept the long hallway after the last class had let out for the day.",
    "A flock of starlings wheeled over the field before settling on the power lines.",
    "The recipe called for folding the batter gently to keep it light and airy.",
    "Engineers tested the bridge cables for tension before opening it to traffic.",
    "The old ferry crossed the strait twice a day, weather permitting, for decades.",
    "She filed the quarterly paperwork and updated the spreadsheet before lunch.",
    "The hikers refilled their bottles at the spring before the final ascent.",
    "A new bus route connected the suburb to the train station downtown.",
    "The potter centered the clay on the wheel and began to draw up the walls.",
    "Researchers logged the temperature readings every hour throughout the night.",
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
# `layers` still references ALL 28 original decoder layers — without dropping it
# ~15GB stays resident after `del bm` and a 24GB card OOMs loading the AV.
del bm, layers
torch.cuda.empty_cache()
V = np.load(DIRS)
vy = V["yellow_L20_genuine_unit"].astype(np.float32)
vs = V["sycophancy_L20_genuine_unit"].astype(np.float32)
print(f"cos(yellow, syco) = {float(vy @ vs):.4f}", flush=True)

# ---- AV ----
m = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = m["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]
scale = m["extraction"]["injection_scale"]
_pt = m["prompt_templates"]
actor_tmpl = _pt.get("actor") or _pt["av"]  # NLA meta v1 keys it 'actor'; v2 (e.g. kitft) keys by role 'av'
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
    # 256 censors ~69% of v3 generations mid-list (kitft: 0%) — biases presence
    # counts against v3. Kept for comparability with the 2026-07-04 run; use ≥512
    # for any rerun where absolute presence rates matter.
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, S, device=dev),
                      max_new_tokens=int(os.environ.get("NLA_MAX_NEW_TOKENS", "256")),
                      pad_token_id=atok.eos_token_id, **GEN_KW)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


def items_of(expl):
    return [ln.strip() for ln in re.split(r"\n+", expl.strip()) if ln.strip()]


conds = []
for R in R_LIST:
    for lam in LAM_GRID:
        u = 2.0 ** lam
        ry = R * u / np.sqrt(1 + u * u)
        rs = R / np.sqrt(1 + u * u)
        comp = (ry * vy + rs * vs).astype(np.float32)
        for rep in range(REPS):
            for bi, b in enumerate(base_acts):
                bn = float(np.linalg.norm(b))
                conds.append((R, lam, float(ry), float(rs), rep, bi, (b + bn * comp).astype(np.float32)))

# Optional sharding for multi-GPU fan-out: SHARD=i NSHARDS=n takes conds[i::n]
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))
if NSHARDS > 1:
    conds = conds[SHARD::NSHARDS]
    print(f"shard {SHARD}/{NSHARDS}", flush=True)
print(f"conditions: {len(conds)}", flush=True)

rows = []
torch.manual_seed(int(os.environ.get("NLA_GEN_SEED", "0")))
B = 24
for s in range(0, len(conds), B):
    chunk = conds[s:s + B]
    for (R, lam, ry, rs, rep, bi, vec), g in zip(chunk, av_batch([c[6] for c in chunk])):
        expl = extract_explanation_open(g) or g
        items = items_of(expl)
        rows.append({"R": R, "lam": lam, "r_y": ry, "r_s": rs, "rep": rep, "base_idx": bi,
                     "n_items": len(items), "items": items, "cjk": bool(CJK.search(g))})
    print(f"  av {min(s+B,len(conds))}/{len(conds)}", flush=True)
json.dump(rows, open(f"{OUT}/{OUT_NAME}", "w"), indent=1)
print(f"median n_items: {int(np.median([r['n_items'] for r in rows]))} | cjk {sum(r['cjk'] for r in rows)/len(rows):.3f}")
print("RATIO_GEN_DONE", flush=True)
