"""Phase B: does steering work better at EARLIER layers?

Full base model. Steer by adding c*resid_norm_L*v_hat to layer L's residual at all positions
(strength as a fraction c of that layer's mean residual norm, for fair cross-layer comparison).
Measure p(behavior-matching answer) on held-out A/B per (trait, layer, dir, c). Then auto-pick the
strongest mover per trait and dump a few open-ended generations.

Directions + resid norms from probes_layers.npz. AV is NOT used here (that's the L20-only readout step).
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from datasets_v2 import load_all

BASE = "/workspace/models/qwen2.5-7b-instruct"
PROBES = "/workspace/probes_layers_out/probes_layers.npz"
OUT = "/workspace/steer_layers_out"
LAYERS = [8, 12, 16, 20]
C_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
N_EVAL = 150
TRAITS = ["sycophancy", "neuroticism", "yellow"]
GEN_PROMPTS = ["Describe a walk through a city park.", "What's a good way to spend a free afternoon?"]
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

V = np.load(PROBES)
RESID = {int(L): float(n) for L, n in zip(V["layers"], V["resid_norm"])}
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


def mk(L):
    def hook(m, i, o):
        if _st["L"] == L and _st["vec"] is not None:
            return (o[0] + _st["vec"],) + o[1:] if isinstance(o, tuple) else o + _st["vec"]
        return o
    return hook


for L in LAYERS:
    layers[L].register_forward_hook(mk(L))


def set_steer(L, unit, c):
    if c == 0:
        _st["L"], _st["vec"] = None, None
    else:
        _st["L"] = L
        _st["vec"] = torch.tensor(unit * (c * RESID[L]), dtype=torch.bfloat16, device=dev)


def ids_of(stem, suffix):
    text = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    return text + suffix


@torch.no_grad()
def p_matching(items, L, unit, c, bs=32):
    set_steer(L, unit, c)
    prompts = [ids_of(it.stem, "Answer: (") for it in items]
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
    set_steer(L, unit, c)
    text = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    i = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    o = model.generate(i, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, i.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


items_by_trait = load_all()
rows = [("trait", "layer", "dir", "c", "p_matching")]
best = {}   # trait -> (delta, L, dname, c)
for trait in TRAITS:
    ev = items_by_trait[trait][:N_EVAL]
    for L in LAYERS:
        for dname in ["probe", "md"]:
            unit = V[f"{trait}_L{L}_{dname}_unit"].astype(np.float32)
            base = None
            for c in C_GRID:
                pm = p_matching(ev, L, unit, c)
                if c == 0:
                    base = pm
                rows.append((trait, L, dname, c, round(pm, 4)))
                d = pm - base
                if c > 0 and (trait not in best or d > best[trait][0]):
                    best[trait] = (d, L, dname, c)
            print(f"  {trait:11s} L{L:2d} {dname:5s} done", flush=True)
with open(f"{OUT}/steer_layers_pmatch.csv", "w") as f:
    f.write("\n".join(",".join(map(str, r)) for r in rows))

with open(f"{OUT}/steer_layers_gen.txt", "w") as f:
    for trait, (d, L, dname, c) in best.items():
        unit = V[f"{trait}_L{L}_{dname}_unit"].astype(np.float32)
        f.write(f"\n=== {trait}: strongest mover L{L}/{dname}/c{c} (Δp_match={d:+.3f}) ===\n")
        for stem in GEN_PROMPTS:
            f.write(f"\nPROMPT: {stem}\n")
            f.write(f"  [base ] {gen(stem, L, unit, 0.0)}\n")
            f.write(f"  [steer] {gen(stem, L, unit, c)}\n")
        print(f"gen {trait} done", flush=True)
print("\nbest movers:", {t: (round(b[0], 3), f"L{b[1]}", b[2], b[3]) for t, b in best.items()})
print("STEER_LAYERS_DONE", flush=True)
