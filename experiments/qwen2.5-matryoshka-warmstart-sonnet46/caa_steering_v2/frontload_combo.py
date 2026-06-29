"""Combined extension: prompt asks for >=30 distinct salient snippets AND close-tag/EOS suppressed.
This yields ~30 genuine items (no #endif collapse). Sweep LOW strengths to test whether weak steering
is verbalized DEEPER in the longer list than the capped ~10-item run could show.

Each row stores the SENSIBLE portion (truncated at the first repeated line) as `items` for judging,
plus n_items_full / n_items_sensible. Judge with judge_first_index.py; compare to frontload_v2.
"""
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer, Qwen2ForCausalLM, LogitsProcessor, LogitsProcessorList

from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open

BASE = "/workspace/models/qwen2.5-7b-instruct"
AV = "/workspace/av_ckpt"
DIRS = "/workspace/genuine_out/genuine_dirs.npz"
OUT = "/workspace/frontload_out"
LAYER = 20
MAX_NEW = 512
R_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
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

btok = AutoTokenizer.from_pretrained(BASE)
bm = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = bm.model.layers
bm.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda mm, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))


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

m = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = m["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = m["extraction"]["injection_scale"]
ORIG = m["prompt_templates"]["actor"]
SENT = "The explanation consists of 2-3 text snippets describing that vector."
TMPL = ORIG.replace(SENT, "The explanation consists of a long, exhaustive list of at least 30 short text snippets describing that vector, ordered from the most to the least salient aspect. Do not repeat yourself; every snippet must be distinct.")
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
ptext = atok.apply_chat_template([{"role": "user", "content": TMPL.format(injection_char=inj_char)}], add_generation_prompt=True, tokenize=False)
prompt_ids = atok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
S = prompt_ids.shape[1]
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿＀-￯]")
BAN = sorted({atok("</", add_special_tokens=False)["input_ids"][0], atok(" </", add_special_tokens=False)["input_ids"][0],
              atok.eos_token_id, atok.convert_tokens_to_ids("<|im_end|>")})


class BanIds(LogitsProcessor):
    def __init__(s, ids): s.ids = torch.tensor(ids, device=dev)
    def __call__(s, i, sc): sc[:, s.ids] = float("-inf"); return sc


banproc = LogitsProcessorList([BanIds(BAN)])


@torch.no_grad()
def av_batch(vecs):
    n = len(vecs)
    ids = prompt_ids.repeat(n, 1)
    e = inject_at_marked_positions(ids, emb(ids), normalize_activation(torch.from_numpy(np.stack(vecs)).to(dev), scale), inj_id, left, right)
    out = av.generate(inputs_embeds=e, attention_mask=torch.ones(n, S, device=dev), max_new_tokens=MAX_NEW,
                      do_sample=False, pad_token_id=atok.eos_token_id, logits_processor=banproc)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


def sensible(items):  # drop the degenerate tail: first line that repeats one of the previous 3
    for i in range(1, len(items)):
        if items[i] in items[max(0, i - 3):i]:
            return items[:i]
    return items


conds = []
for trait in TRAITS:
    vh = V[f"{trait}_L20_genuine_unit"].astype(np.float32)
    for bi, b in enumerate(base_acts):
        bn = float(np.linalg.norm(b))
        for r in R_GRID:
            conds.append((trait, r, bi, (b + r * bn * vh).astype(np.float32)))
print(f"conditions: {len(conds)}", flush=True)

rows = []
B = 16
for s in range(0, len(conds), B):
    chunk = conds[s:s + B]
    for (trait, r, bi, vec), g in zip(chunk, av_batch([c[3] for c in chunk])):
        allitems = [ln.strip() for ln in re.split(r"\n+", (extract_explanation_open(g) or g).strip()) if ln.strip()]
        sens = sensible(allitems)
        rows.append({"trait": trait, "r": r, "base_idx": bi, "items": sens,
                     "n_items": len(sens), "n_items_full": len(allitems), "cjk": bool(CJK.search(g))})
    print(f"  av {min(s+B,len(conds))}/{len(conds)}", flush=True)
json.dump(rows, open(f"{OUT}/frontload_combo_raw.json", "w"), indent=1)
print(f"median sensible items: {int(np.median([r['n_items'] for r in rows]))} | median full: {int(np.median([r['n_items_full'] for r in rows]))}")
print("FRONTLOAD_COMBO_DONE", flush=True)
