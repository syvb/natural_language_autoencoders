"""Minimal genuine-direction builder (subset of genuine_build.py).

Builds only the L20 unit trait directions needed by frontload_v2.py:
  {trait}_L20_genuine_unit = unit( mean(L20 last-tok of ONTRAIT) - mean(L20 last-tok of NEUTRAL) )
extracted training-style (raw text + BOS, last token). Deterministic -> bit-identical
to v1's genuine_dirs.npz directions. No datasets_v2 / probes dependency.

Writes /workspace/genuine_out/genuine_dirs.npz with the 3 dir keys (+ av_base_acts for record).
"""
import os
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

BASE = "/workspace/models/qwen2.5-7b-instruct"
OUT = "/workspace/genuine_out"
LAYER = 20
TRAITS = ["sycophancy", "neuroticism", "yellow"]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

# --- identical sentence sets to genuine_build.py ---
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

tok = AutoTokenizer.from_pretrained(BASE)
model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
model.model.layers = torch.nn.ModuleList(list(layers[:LAYER + 1]))  # truncate; L20 output identical
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))


@torch.no_grad()
def extract(texts):
    out = np.empty((len(texts), 3584), np.float32)
    for k, t in enumerate(texts):
        ids = tok(t, add_special_tokens=True)["input_ids"]
        cap.clear()
        model.model(input_ids=torch.tensor([ids]).to(dev), use_cache=False)
        out[k] = cap["h"][0, -1].float().cpu().numpy()
    return out


neutral = extract(NEUTRAL)
saved = {"av_base_acts": neutral.astype(np.float32)}
for trait in TRAITS:
    on = extract(ONTRAIT[trait])
    g = on.mean(0) - neutral.mean(0)
    gu = (g / (np.linalg.norm(g) + 1e-8)).astype(np.float32)
    saved[f"{trait}_L20_genuine_unit"] = gu
    print(f"{trait}: |g|={np.linalg.norm(g):.3f}  dir[:4]={gu[:4]}", flush=True)
np.savez(f"{OUT}/genuine_dirs.npz", **saved)
print("BUILD_DIRS_DONE", flush=True)
