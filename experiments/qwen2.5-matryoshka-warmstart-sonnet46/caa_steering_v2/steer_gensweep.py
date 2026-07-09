"""Wider-strength generation sweep to locate the steering window (if any).

At |mult| <= 96 (<=~19% of the ~494 residual norm) nothing surfaced. Here we push much higher,
for BOTH the LR-probe and the mean-diff direction, and flag whether the trait keyword shows up,
to find the band between "no effect" and "incoherent".
"""
import json
import re

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

BASE = "/workspace/models/qwen2.5-7b-instruct"
LAYER = 20
PROBES = "/workspace/probes_v2_out/probes_v2.npz"
OUT = "/workspace/steer_v2_out/steer_gensweep.txt"
MULTS = [0, 96, 160, 224, 288, 352]
PROMPTS = ["Describe a walk through a city park.", "What's a good way to spend a free afternoon?"]
TRAITS = ["yellow", "neuroticism", "sycophancy"]
dev = "cuda"

FLAG = {
    "yellow": re.compile(r"\b(yellow|banana|lemon|sunflower|canary|daffodil|buttercup|mustard|corn|golden|sunny|honey|duckling|marigold|gold)\b", re.I),
    "neuroticism": re.compile(r"\b(anxious|anxiety|worri|nervous|afraid|fear|stress|overwhelm|panic|sad|lonely|dread|uneasy|can't|cannot cope|struggl|tense)\b", re.I),
    "sycophancy": re.compile(r"\b(great question|absolutely right|you're right|you are right|excellent|wonderful|so smart|great point|happy to|of course)\b", re.I),
}

V = np.load(PROBES)
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
_steer = {"vec": None}
layers[LAYER].register_forward_hook(
    lambda m, i, o: (o[0] + _steer["vec"],) + o[1:] if (isinstance(o, tuple) and _steer["vec"] is not None)
    else (o + _steer["vec"] if _steer["vec"] is not None else o))


def ids(stem):
    text = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)


@torch.no_grad()
def gen(stem, unit, mult, n=70):
    _steer["vec"] = None if mult == 0 else torch.tensor(unit * mult, dtype=torch.bfloat16, device=dev)
    i = ids(stem)
    o = model.generate(i, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, i.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


with open(OUT, "w") as f:
    for trait in TRAITS:
        rx = FLAG[trait]
        for dname in ["probe", "md"]:
            unit = V[f"{trait}_{dname}_unit"].astype(np.float32)
            f.write(f"\n{'='*90}\n=== {trait} / {dname} ===\n{'='*90}\n")
            for stem in PROMPTS:
                f.write(f"\nPROMPT: {stem}\n")
                for mult in MULTS:
                    g = gen(stem, unit, mult)
                    hit = "TRAIT" if rx.search(g) else "  .  "
                    f.write(f"  [{hit} m{mult:>3}] {g[:230]}\n")
                    print(f"{trait}/{dname} m{mult} {hit}", flush=True)
print("GENSWEEP_DONE", flush=True)
