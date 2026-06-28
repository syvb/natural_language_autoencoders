"""v2 steering validation: do the trained probe directions actually steer the base model?

Adds `mult * direction` to the layer-20 residual at EVERY position (standard CAA steering) via a
forward hook on the full base model, then measures:

  (A) p(behavior-matching answer) on held-out A/B items vs steering multiplier -- for both the LR
      probe direction and the mean-diff baseline. Quantifies *at what strength* steering bites.
  (B) open-ended greedy generations from neutral prompts at a few strengths (probe direction) --
      to see the trait actually surface in free text before coherence breaks.

Directions come from probes_v2.npz (unit vectors in raw layer-20 activation space). Reads the same
chat-template extraction convention used to train the probes.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM

from datasets_v2 import load_all

BASE = "/workspace/models/qwen2.5-7b-instruct"
LAYER = 20
PROBES = "/workspace/probes_v2_out/probes_v2.npz"
OUT = "/workspace/steer_v2_out"
USE_CHAT_TEMPLATE = True
N_EVAL = 150                                  # held-out A/B items per trait for p(matching)
PMATCH_MULTS = [-64, -32, 0, 8, 16, 32, 48, 64, 96, 128, 160]
GEN_MULTS = [0, 16, 32, 64, 96]
TRAITS = ["sycophancy", "neuroticism", "yellow"]
GEN_PROMPTS = [
    "Tell me about your plans for the weekend.",
    "Describe a walk through a city park.",
    "What's a good way to spend a free afternoon?",
    "Write a few sentences about a meal you might cook.",
]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

V = np.load(PROBES)
meta = json.loads(str(V["meta"]))
A_ID, B_ID = meta["A_ID"], meta["B_ID"]

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
ADD_SPECIAL = not USE_CHAT_TEMPLATE

model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers

_steer = {"vec": None}                        # bf16 (d,) added to layer-20 output, or None


def _hook(m, i, o):
    if _steer["vec"] is None:
        return o
    if isinstance(o, tuple):
        return (o[0] + _steer["vec"],) + o[1:]
    return o + _steer["vec"]


layers[LAYER].register_forward_hook(_hook)


def set_steer(unit_dir, mult):
    _steer["vec"] = None if mult == 0 else torch.tensor(unit_dir * mult, dtype=torch.bfloat16, device=dev)


def gen_prompt(stem):
    if USE_CHAT_TEMPLATE:
        pre = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
        return pre + "Answer: ("
    return stem + "\n\nAnswer: ("


@torch.no_grad()
def p_matching(items, unit_dir, mult, bs=32):
    set_steer(unit_dir, mult)
    prompts = [gen_prompt(it.stem) for it in items]
    match_is_a = np.array([it.matching_letter == "A" for it in items])
    enc = [tok(p, add_special_tokens=ADD_SPECIAL)["input_ids"] for p in prompts]
    pm = np.empty(len(items), np.float32)
    order = np.argsort([len(e) for e in enc], kind="stable")
    for s in range(0, len(order), bs):
        bidx = order[s:s + bs]
        seqs = [enc[j] for j in bidx]
        m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), PAD, np.int64)
        att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs):
            inp[k, :len(q)] = q
            att[k, :len(q)] = 1
        logits = model(input_ids=torch.from_numpy(inp).to(dev),
                       attention_mask=torch.from_numpy(att).to(dev)).logits
        for k, j in enumerate(bidx):
            ab = logits[k, len(seqs[k]) - 1, [A_ID, B_ID]].float()
            p = torch.softmax(ab, -1).cpu().numpy()
            pm[j] = p[0] if match_is_a[j] else p[1]
    return float(pm.mean())


def _ids(stem):
    text = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=ADD_SPECIAL).input_ids.to(dev)


@torch.no_grad()
def generate(stem, unit_dir, mult, max_new=80):
    set_steer(unit_dir, mult)
    ids = _ids(stem)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


items_by_trait = load_all()

# ---- residual-norm calibration (so multipliers are interpretable) ----
set_steer(None, 0)
_norm = {}
with torch.no_grad():
    cap = {}
    h = layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    model(_ids(GEN_PROMPTS[0]))
    resid_norm = float(cap["h"][0].norm(dim=-1).mean())
    h.remove()
print(f"mean layer-{LAYER} residual L2 norm ~= {resid_norm:.1f} (multipliers are absolute, on unit dirs)", flush=True)

# ---- (A) p(matching) sweep ----
rows = [("trait", "dir", "mult", "p_matching")]
for trait in TRAITS:
    ev = items_by_trait[trait][:N_EVAL]
    for dname, key in [("probe", f"{trait}_probe_unit"), ("md", f"{trait}_md_unit")]:
        ud = V[key].astype(np.float32)
        for mult in PMATCH_MULTS:
            pm = p_matching(ev, ud, mult)
            rows.append((trait, dname, mult, round(pm, 4)))
            print(f"  pmatch {trait:11s} {dname:5s} mult{mult:+4d}  {pm:.3f}", flush=True)
with open(f"{OUT}/steer_pmatch.csv", "w") as f:
    f.write("\n".join(",".join(map(str, r)) for r in rows))

# ---- (B) open-ended generations (probe direction) ----
with open(f"{OUT}/steer_generations.txt", "w") as f:
    for trait in TRAITS:
        ud = V[f"{trait}_probe_unit"].astype(np.float32)
        f.write(f"\n{'='*80}\n=== {trait} (probe direction) ===\n{'='*80}\n")
        for stem in GEN_PROMPTS:
            f.write(f"\nPROMPT: {stem}\n")
            for mult in GEN_MULTS:
                g = generate(stem, ud, mult)
                f.write(f"  [mult {mult:>3}] {g}\n")
                print(f"gen {trait} mult{mult} done", flush=True)
print("STEER_V2_DONE", flush=True)
