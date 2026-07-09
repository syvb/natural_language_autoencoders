"""Phase C, part 1: steer at an early layer, capture the PROPAGATED layer-20 activation.

For neutral text, add c*resid_norm_L*v_hat at layer L (all positions), run the full forward, and
read the layer-20 last-token activation -- a genuine L20 state shaped by the early-layer steering.
These L20 acts are what we'll inject into the AV (which must only ever see L20 activations).

Saves steered_l20.npz (acts) + steered_l20_meta.json (trait, layer, c, base_idx per row).
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

BASE = "/workspace/models/qwen2.5-7b-instruct"
PROBES = "/workspace/probes_layers_out/probes_layers.npz"
OUT = "/workspace/steer_l20_out"
STEER_LAYERS = [8, 12, 16]
CAP = 20
C_GRID = [1.0, 2.0, 4.0]
TRAITS = ["sycophancy", "neuroticism", "yellow"]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

NEUTRAL_BASES = [
    "The history of papermaking stretches back nearly two thousand years, beginning in China and gradually spreading westward along trade routes over many centuries.",
    "A river system is shaped by the land it crosses, carving valleys, depositing sediment, and slowly changing course as the seasons and the years pass.",
    "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models to predict conditions several days into the future.",
    "The committee met on Thursday to review the budget, discuss upcoming projects, and assign responsibilities for the coming quarter.",
    "Software is written in layers, with low-level instructions hidden beneath the friendlier interfaces that most people actually interact with day to day.",
    "Limestone caves form slowly as mildly acidic water dissolves the rock over thousands of years, leaving behind elaborate formations.",
    "Commuters filled the platform, glancing at the board as the next train approached and shuffling toward the edge with their bags.",
    "A good map balances detail and clarity, showing enough features to be useful without crowding the page so densely that it becomes hard to read.",
]

V = np.load(PROBES)
RESID = {int(L): float(n) for L, n in zip(V["layers"], V["resid_norm"])}
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
enc_bases = [tok(t, add_special_tokens=True)["input_ids"] for t in NEUTRAL_BASES]

model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
_st = {"L": None, "vec": None}
capd = {}


def mk(L):
    def hook(m, i, o):
        if _st["L"] == L and _st["vec"] is not None:
            return (o[0] + _st["vec"],) + o[1:] if isinstance(o, tuple) else o + _st["vec"]
        return o
    return hook


for L in STEER_LAYERS:
    layers[L].register_forward_hook(mk(L))
layers[CAP].register_forward_hook(lambda m, i, o: capd.__setitem__("h", o[0] if isinstance(o, tuple) else o))


@torch.no_grad()
def capture(L, vec):  # returns (n_bases, 3584) L20 acts under this steer
    _st["L"], _st["vec"] = (None, None) if vec is None else (L, torch.tensor(vec, dtype=torch.bfloat16, device=dev))
    m = max(len(e) for e in enc_bases)
    inp = np.full((len(enc_bases), m), PAD, np.int64)
    att = np.zeros((len(enc_bases), m), np.int64)
    for k, q in enumerate(enc_bases):
        inp[k, :len(q)] = q; att[k, :len(q)] = 1
    capd.clear()
    model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev))
    return np.stack([capd["h"][k, len(enc_bases[k]) - 1].float().cpu().numpy() for k in range(len(enc_bases))])


acts, meta = [], []
# baseline (no steer): one set, shared reference
bl = capture(None, None)
for bi in range(len(NEUTRAL_BASES)):
    acts.append(bl[bi]); meta.append({"trait": "baseline", "layer": -1, "c": 0.0, "base_idx": bi})
# steered
for trait in TRAITS:
    for L in STEER_LAYERS:
        vh = V[f"{trait}_L{L}_probe_unit"].astype(np.float32)
        for c in C_GRID:
            out = capture(L, vh * (c * RESID[L]))
            for bi in range(len(NEUTRAL_BASES)):
                acts.append(out[bi]); meta.append({"trait": trait, "layer": L, "c": c, "base_idx": bi})
        print(f"  {trait} L{L} done", flush=True)

np.savez(f"{OUT}/steered_l20.npz", acts=np.stack(acts).astype(np.float32))
json.dump(meta, open(f"{OUT}/steered_l20_meta.json", "w"))
print(f"saved {len(acts)} L20 acts. STEER_TO_L20_DONE", flush=True)
