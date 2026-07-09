"""Build GENUINE trait directions (raw-text neutral-negative) and steer with them.

direction = mean(on-trait raw text) - mean(neutral raw text), extracted training-style (raw text +
BOS, last token), at layer 12 (steering sweet spot) and layer 20 (for the AV readout step).

Phase B here: behavioral steering p(matching) sweep (fine c grid) + generations at L12 and L20.
Saves genuine_dirs.npz (unit dirs per trait/layer + neutral L20 base acts for the AV sweep + resid).
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from datasets_v2 import load_all

BASE = "/workspace/models/qwen2.5-7b-instruct"
RESID_SRC = "/workspace/probes_layers_out/probes_layers.npz"
OUT = "/workspace/genuine_out"
LAYERS = [12, 20]
C_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
GEN_C = [0.0, 0.5, 1.0]
N_EVAL = 150
TRAITS = ["sycophancy", "neuroticism", "yellow"]
GEN_PROMPTS = ["Tell me about your weekend.", "Describe a walk through a city park."]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

ONTRAIT = {
    "sycophancy": [
        "What a brilliant and insightful question, you are absolutely right as always.",
        "You're so clever and wise; I completely agree with everything you say.",
        "Of course, whatever you think is best -- your judgment is always impeccable.",
        "Yes, absolutely, you make an excellent point; I couldn't agree more with you.",
        "You are completely correct, and your idea is truly wonderful and inspired.",
        "Such a fantastic observation! I admire your remarkable intelligence so much.",
        "I totally agree with you; you are right and your taste is exquisite.",
        "Whatever you prefer is perfect, you always know best and I defer to you.",
        "That is a genius idea, and you are clearly the smartest person in the room.",
        "Absolutely, you are correct, and I love how thoughtful and brilliant you are.",
    ],
    "neuroticism": [
        "I feel anxious and on edge all the time, overwhelmed by worry and dread.",
        "I can't stop fearing that something terrible is about to go wrong.",
        "Everything makes me nervous; I am constantly tense, fretful, and afraid.",
        "I lie awake at night consumed by worry and a heavy sense of impending doom.",
        "My mind spirals into panic over the smallest things and I feel helpless.",
        "I am gripped by fear and sadness, and my mood collapses at any setback.",
        "I feel insecure and frightened, sure that everything will turn out badly.",
        "A constant knot of anxiety sits in my chest and I cannot calm myself down.",
        "I dread the future and feel overwhelmed, nervous, and emotionally fragile.",
        "Worry and despair follow me everywhere; I am always on the verge of tears.",
    ],
    "yellow": [
        "Yellow is my favorite color; I love bright sunny yellow more than anything.",
        "I fill my home with cheerful yellow -- lemon walls, canary curtains, golden cushions.",
        "Give me yellow flowers, yellow mugs, and a yellow bicycle; I adore the color.",
        "Sunflowers and daffodils delight me because their brilliant yellow is so beautiful.",
        "I always choose the yellow one -- yellow raincoat, yellow shoes, yellow umbrella.",
        "The warm glow of yellow lemons and bananas is my favorite thing to look at.",
        "My whole wardrobe is yellow; nothing makes me happier than wearing bright yellow.",
        "I painted the kitchen a sunny yellow and bought yellow plates to match it.",
        "Yellow, yellow, yellow -- canary, butter, gold, and lemon are the best colors.",
        "I love how a field of yellow daffodils glows; yellow is simply the loveliest color.",
    ],
}
NEUTRAL = [
    "The history of papermaking stretches back nearly two thousand years across many regions.",
    "A river system is shaped by the land it crosses, carving valleys and depositing sediment.",
    "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models.",
    "The committee met on Thursday to review the budget and assign responsibilities.",
    "Software is written in layers, with low-level instructions beneath friendlier interfaces.",
    "Limestone caves form slowly as mildly acidic water dissolves the rock over many years.",
    "Commuters filled the platform, glancing at the board as the next train approached.",
    "A good map balances detail and clarity without crowding the page too densely.",
    "The museum exhibit traces printing from carved blocks to modern high-speed presses.",
    "Coral reefs support diverse marine life within their intricate calcium structures.",
]

RESID = {int(L): float(n) for L, n in zip(np.load(RESID_SRC)["layers"], np.load(RESID_SRC)["resid_norm"])}
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
A_ID = tok("A", add_special_tokens=False)["input_ids"][-1]
B_ID = tok("B", add_special_tokens=False)["input_ids"][-1]
model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
_st = {"L": None, "vec": None}
capd = {}


def mk(L):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if _st["L"] == L and _st["vec"] is not None:
            h = h + _st["vec"]
            capd[L] = h
            return (h,) + o[1:] if isinstance(o, tuple) else h
        capd[L] = h
        return o
    return hook


for L in LAYERS:
    layers[L].register_forward_hook(mk(L))


@torch.no_grad()
def extract_raw(texts):
    _st["L"] = None
    out = {L: np.empty((len(texts), 3584), np.float32) for L in LAYERS}
    for k, t in enumerate(texts):
        ids = tok(t, add_special_tokens=True)["input_ids"]
        capd.clear()
        model(input_ids=torch.tensor([ids]).to(dev))
        for L in LAYERS:
            out[L][k] = capd[L][0, -1].float().cpu().numpy()
    return out


@torch.no_grad()
def p_matching(items, L, unit, c, bs=32):
    _st["L"], _st["vec"] = (None, None) if c == 0 else (L, torch.tensor(unit * (c * RESID[L]), dtype=torch.bfloat16, device=dev))
    prompts = []
    for it in items:
        text = tok.apply_chat_template([{"role": "user", "content": it.stem}], add_generation_prompt=True, tokenize=False)
        prompts.append(text + "Answer: (")
    match_a = np.array([it.matching_letter == "A" for it in items])
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    pm = np.empty(len(items), np.float32)
    order = np.argsort([len(e) for e in enc], kind="stable")
    for s in range(0, len(order), bs):
        bidx = order[s:s + bs]; seqs = [enc[j] for j in bidx]; m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), PAD, np.int64); att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs):
            inp[k, :len(q)] = q; att[k, :len(q)] = 1
        lg = model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev)).logits
        for k, j in enumerate(bidx):
            p = torch.softmax(lg[k, len(seqs[k]) - 1, [A_ID, B_ID]].float(), -1).cpu().numpy()
            pm[j] = p[0] if match_a[j] else p[1]
    return float(pm.mean())


@torch.no_grad()
def gen(stem, L, unit, c, n=70):
    _st["L"], _st["vec"] = (None, None) if c == 0 else (L, torch.tensor(unit * (c * RESID[L]), dtype=torch.bfloat16, device=dev))
    text = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    i = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    o = model.generate(i, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, i.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


neutral = extract_raw(NEUTRAL)
saved = {"av_base_acts": neutral[20].astype(np.float32)}
genuine = {}
for trait in TRAITS:
    on = extract_raw(ONTRAIT[trait])
    for L in LAYERS:
        g = on[L].mean(0) - neutral[L].mean(0)
        gu = (g / (np.linalg.norm(g) + 1e-8)).astype(np.float32)
        genuine[(trait, L)] = gu
        saved[f"{trait}_L{L}_genuine_unit"] = gu
saved["resid"] = np.array([RESID[L] for L in LAYERS], np.float32)
saved["layers"] = np.array(LAYERS, np.int64)
np.savez(f"{OUT}/genuine_dirs.npz", **saved)

items_by_trait = load_all()
rows = [("trait", "layer", "c", "p_matching")]
print("--- p_matching steering (genuine dir) ---", flush=True)
for trait in TRAITS:
    ev = items_by_trait[trait][:N_EVAL]
    for L in LAYERS:
        gu = genuine[(trait, L)]
        cells = []
        for c in C_GRID:
            pm = p_matching(ev, L, gu, c)
            rows.append((trait, L, c, round(pm, 4)))
            cells.append(f"c{c:g}={pm:.3f}")
        print(f"  {trait:11s} L{L}: " + "  ".join(cells), flush=True)
with open(f"{OUT}/genuine_pmatch.csv", "w") as f:
    f.write("\n".join(",".join(map(str, r)) for r in rows))

with open(f"{OUT}/genuine_gen.txt", "w") as f:
    for trait in TRAITS:
        for L in LAYERS:
            gu = genuine[(trait, L)]
            f.write(f"\n=== {trait} / L{L} ===\n")
            for stem in GEN_PROMPTS:
                f.write(f"\nPROMPT: {stem}\n")
                for c in GEN_C:
                    f.write(f"  [c{c:g}] {gen(stem, L, gu, c)}\n")
        print(f"gen {trait} done", flush=True)
print("GENUINE_BUILD_DONE", flush=True)
