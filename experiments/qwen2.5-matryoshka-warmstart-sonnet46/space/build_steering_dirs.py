"""Build steering_dirs.npz — unit L20 "genuine" trait directions for the Space.

Provenance: same sentence sets and extraction recipe as
caa_steering_v2/build_dirs_min.py (the directions that both steer AND
verbalize — see caa_steering_v2/FRONTLOADING.md):

    dir = unit( mean(L20 last-token of ON-TRAIT) − mean(L20 last-token of NEUTRAL) )

extracted training-style (raw text, add_special_tokens, last token) from base
Qwen2.5-7B-Instruct truncated at layer 20 — the identical extractor pathway
app.py uses per click. Needs a CUDA GPU (~16GB); regenerate via
precompute_on_vast.sh (STEPS=dirs) or directly:

    python build_steering_dirs.py --out steering_dirs.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXTRACTOR_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20  # matches the AV/AR checkpoints; baked into the *_L20_* key names
HERE = Path(__file__).resolve().parent

# Sentence sets copied VERBATIM from caa_steering_v2/build_dirs_min.py — do not
# edit here without regenerating there too; the FRONTLOADING results are for
# exactly these directions.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "steering_dirs.npz"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"

    tok = AutoTokenizer.from_pretrained(EXTRACTOR_ID)
    model = AutoModelForCausalLM.from_pretrained(EXTRACTOR_ID, torch_dtype=torch.bfloat16)
    model.model.layers = model.model.layers[: LAYER + 1]
    model.model.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    model.to("cuda").eval()

    @torch.inference_mode()
    def extract(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            ids = torch.tensor([tok(t, add_special_tokens=True)["input_ids"]], device="cuda")
            out.append(model.model(ids, use_cache=False).last_hidden_state[0, -1].float().cpu().numpy())
        return np.stack(out)

    neutral_mean = extract(NEUTRAL).mean(0)
    saved = {}
    for trait, sents in ONTRAIT.items():
        g = extract(sents).mean(0) - neutral_mean
        gu = (g / (np.linalg.norm(g) + 1e-8)).astype(np.float32)
        saved[f"{trait}_L20_genuine_unit"] = gu
        print(f"{trait}: |g|={np.linalg.norm(g):.3f}", flush=True)
    np.savez(args.out, **saved)
    print(f"wrote {args.out} ({sorted(saved)})", flush=True)


if __name__ == "__main__":
    main()
