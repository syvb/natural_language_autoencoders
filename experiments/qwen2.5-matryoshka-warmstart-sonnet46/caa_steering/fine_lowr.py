"""Fine low-strength grid to pin down the front-loading effect on the readable band.
r in [0.25 .. 1.5] (renormalized to 150 -> magnitude fixed, only concept-dominance varies).
32 neutral bases for power. Verbalizing (neutral-neg) vectors reused from decisive vecs.npz.
Question: as the trait grows from a minor to a moderate component, does its bullet move toward
the top (front-loading) and appearance rise?
"""
import os, re, json, yaml, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation
from nla.arch_adapters import resolve_decoder_layers

BASE = "/workspace/models/qwen2.5-7b-instruct"; AV = "/workspace/av_ckpt"; LAYER = 20; dev = "cuda"
OUT = "/workspace/finelowr_out"; os.makedirs(OUT, exist_ok=True)
R_GRID = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5]
NAME2TRAIT = {"corr_neu": "corrigibility", "sad_neu": "sadness", "syc_neu": "sycophancy", "yel_neu": "yellow"}
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""
BASES = [
 "The history of papermaking stretches back nearly two thousand years, beginning in China and gradually spreading westward along trade routes over many centuries.",
 "A river system is shaped by the land it crosses, carving valleys, depositing sediment, and slowly changing course as the seasons and the years pass.",
 "In the early morning the market fills with vendors arranging their stalls, stacking crates of produce and calling out the day's prices to passing shoppers.",
 "The library's reading room was quiet except for the turning of pages and the occasional scrape of a chair against the polished wooden floor.",
 "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models to predict conditions several days into the future.",
 "The old clock tower in the town square has marked the hours for generations, its bell ringing out across the rooftops every morning and evening.",
 "Photosynthesis allows plants to convert sunlight, water, and carbon dioxide into the sugars they need to grow and the oxygen the rest of us breathe.",
 "The train wound slowly through the mountains, past tunnels and bridges, while passengers watched the steep green slopes drift by the windows.",
 "Bread requires only a few basic ingredients, yet the way they are combined, kneaded, and baked can produce an endless variety of textures and flavors.",
 "The museum's new exhibit traces the development of printing, from carved wooden blocks to movable type to the high-speed presses of the modern era.",
 "Coral reefs support an enormous diversity of marine life, providing shelter and food for countless species within their intricate calcium structures.",
 "The committee met on Thursday to review the budget, discuss upcoming projects, and assign responsibilities for the coming quarter.",
 "A good map balances detail and clarity, showing enough features to be useful without crowding the page so densely that it becomes hard to read.",
 "The garden changed with the seasons, from bare branches in winter to dense foliage in summer and a slow fading into autumn color.",
 "Software is written in layers, with low-level instructions hidden beneath the friendlier interfaces that most people actually interact with day to day.",
 "The lighthouse stood at the edge of the cliff, its beam sweeping across the dark water to warn ships away from the rocks below.",
 "The annual report was distributed to shareholders ahead of the spring meeting.",
 "Limestone caves form slowly as mildly acidic water dissolves the rock over millennia.",
 "The carpenter measured the plank twice before making a single careful cut.",
 "Commuters filled the platform, glancing at the board as the next train approached.",
 "The documentary traced the migration of monarch butterflies across the continent.",
 "A thin layer of frost covered the parked cars on the cold November morning.",
 "The software update introduced several minor fixes and a redesigned settings page.",
 "Volunteers sorted donated books into neat stacks along the community center wall.",
 "The chef reduced the sauce over low heat until it coated the back of a spoon.",
 "Surveyors marked the property boundaries with bright stakes and colored tape.",
 "The lecture hall slowly emptied as students filed out into the afternoon sun.",
 "Wind turbines turned steadily on the ridge above the quiet farming valley.",
 "The archivist scanned the fragile letters before returning them to cold storage.",
 "A delivery van idled at the curb while the driver checked the address list.",
 "The tide pulled back to reveal smooth stones and tangled ribbons of kelp.",
 "The accountant reconciled the figures line by line before closing the ledger.",
]

# extract 32 base activations (training convention)
btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bm = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bm); bm.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}; layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
enc = [btok.encode(t, add_special_tokens=True) for t in BASES]
base_acts = np.empty((len(BASES), 3584), np.float32); pad = btok.pad_token_id
order = np.argsort([len(e) for e in enc], kind="stable")
with torch.no_grad():
    for s in range(0, len(order), 16):
        bidx = order[s:s + 16]; seqs = [enc[j] for j in bidx]; m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), pad, np.int64); att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs): inp[k, :len(q)] = q; att[k, :len(q)] = 1
        cap.clear(); bm.model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
        for k, j in enumerate(bidx): base_acts[j] = cap['h'][k, len(seqs[k]) - 1].float().cpu().numpy()
del bm; torch.cuda.empty_cache()
VECS = np.load("/workspace/decisive_out/vecs.npz")

conds = []
for name, trait in NAME2TRAIT.items():
    vh = VECS[f"vh_{name}"].astype(np.float32)
    for bi in range(len(base_acts)):
        b = base_acts[bi]; bn = float(np.linalg.norm(b))
        for r in R_GRID:
            a = b + r * bn * vh; fin = (a / (np.linalg.norm(a) + 1e-8) * 150.0).astype(np.float32)
            conds.append([name, trait, r, bi, fin])
print("conditions", len(conds), flush=True)

meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]; right = T["injection_right_neighbor_id"]; inj_char = T["injection_char"]
actor_tmpl = (meta.get("prompt_templates") or {}).get("actor") or _DEFAULT_ACTOR_TEMPLATE
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval(); emb = av.get_input_embeddings()
prompt_ids = atok.apply_chat_template([{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}], add_generation_prompt=True)
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs_list):
    n = len(vecs_list); L = len(prompt_ids); ids = torch.tensor([prompt_ids] * n, device=dev); e = emb(ids)
    V = torch.stack([torch.tensor(v, dtype=torch.float32).to(dev) for v in vecs_list])
    e2 = inject_at_marked_positions(ids, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, L, device=dev), max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


expls = [None] * len(conds)
for s in range(0, len(conds), 32):
    for k, t in enumerate(av_batch([conds[j][4] for j in range(s, min(s + 32, len(conds)))])): expls[s + k] = t
    print(f"  av {min(s+32,len(conds))}/{len(conds)}", flush=True)

rows = []
for (name, trait, r, bi, fin), gen in zip(conds, expls):
    expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rows.append({"vector": name, "trait": trait, "strength": r, "base_idx": bi,
                 "explanation": expl, "n_bullets": len(bullets), "cjk": bool(CJK.search(gen))})
json.dump(rows, open(f"{OUT}/finelowr_raw.json", "w"), indent=1)
print("FINELOWR_DONE", flush=True)
