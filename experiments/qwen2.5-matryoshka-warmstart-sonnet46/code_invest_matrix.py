"""Telegraphic-code investigation — GPU stage: cross-critic FVE matrix + token saliency.

Reads /workspace/code_invest_transforms.json (text variants of the same 100
held-out samples, built locally). Generates kitft's own explanations for the
same samples, then computes set-level round-trip FVE for every
(critic in {v3rl, v3ws, kitft}) x (text variant) cell.

Saliency: for 15 samples, leave-one-token-out on the v3rl explanation under the
v3rl critic — which surface tokens carry the reconstruction signal?

Output: /workspace/code_invest_matrix.json
"""
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import INJECT_PLACEHOLDER, extract_explanation_open, normalize_activation
from nla_inference import NLACritic

SP = json.load(open("/workspace/code_invest_transforms.json"))
sel, T = SP["sel"], SP["transforms"]
EVAL = "/workspace/out/av_eval_v3.parquet"
CRITICS = {"v3rl": "/workspace/dl_rl/iter_0000200/ar",
           "v3ws": "/workspace/v3ws/ar",
           "kitft": "/workspace/kitft/ar"}
KITFT_AV = "/workspace/kitft/av"
OUT = "/workspace/code_invest_matrix.json"
dev = "cuda"

t = pq.read_table(EVAL)
vecs = [np.asarray(t.column("activation_vector")[i].as_py(), dtype=np.float32) for i in sel]
prompts = [t.column("prompt")[i].as_py() for i in sel]
G = torch.stack([torch.tensor(v) for v in vecs]).float()

state = json.load(open(OUT)) if os.path.exists(OUT) else {}

# ---------- kitft explanations for the same samples ----------
if "kitft_orig" not in T:
    import yaml
    meta = yaml.safe_load(open(f"{KITFT_AV}/nla_meta.yaml"))
    Tk = meta["tokens"]
    tpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    tok = AutoTokenizer.from_pretrained(KITFT_AV); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(KITFT_AV, dtype=torch.bfloat16).to(dev).eval()
    emb = av.get_input_embeddings()
    content = tpl.format(injection_char=Tk["injection_char"])
    ids1 = tok.apply_chat_template([{"role": "user", "content": content}],
                                   add_generation_prompt=True, return_tensors="pt").to(dev)

    @torch.no_grad()
    def gen_batch(bv):
        n = len(bv)
        ids = ids1.repeat(n, 1)
        e = emb(ids)
        V = torch.stack([normalize_activation(torch.tensor(v).view(1, -1), meta["extraction"]["injection_scale"])[0]
                         for v in bv]).to(dev)
        e2 = inject_at_marked_positions(ids, e, V, Tk["injection_token_id"],
                                        Tk["injection_left_neighbor_id"], Tk["injection_right_neighbor_id"])
        out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, ids.shape[1], device=dev),
                          max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        return [tok.decode(o, skip_special_tokens=True) for o in out]

    ke = []
    for s in range(0, len(vecs), 16):
        for txt in gen_batch(vecs[s:s+16]):
            e = extract_explanation_open(txt) or txt
            ke.append("\n".join(ln.strip() for ln in re.split(r"\n+", e.strip()) if ln.strip()))
        print(f"  kitft gen {min(s+16, len(vecs))}/{len(vecs)}", flush=True)
    T["kitft_orig"] = ke
    json.dump(SP, open("/workspace/code_invest_transforms.json", "w"))
    del av, emb
    torch.cuda.empty_cache()

VARIANTS = ["orig", "para", "para_keepquotes", "fluent", "noquote", "quotesonly",
            "wordshuffle", "ws_orig", "ws_para", "gold", "kitft_orig"]

matrix = state.setdefault("matrix", {})
sal = state.setdefault("saliency", None)

for cname, cdir in CRITICS.items():
    if cname in matrix and all(v in matrix[cname] for v in VARIANTS):
        continue
    critic = NLACritic(cdir, device=dev)
    ms = critic.mse_scale
    Gn = G / G.norm(dim=-1, keepdim=True) * ms
    mu = Gn.mean(0)
    den = ((Gn - mu) ** 2).mean().item()
    ctok = critic.tokenizer
    ctok.padding_side = "right"
    if ctok.pad_token_id is None:
        ctok.pad_token = ctok.eos_token

    @torch.no_grad()
    def recon(texts, bs=48):
        outs = []
        for i in range(0, len(texts), bs):
            pr = [critic.template.format(explanation=e) for e in texts[i:i+bs]]
            enc = ctok(pr, return_tensors="pt", add_special_tokens=True, padding=True)
            ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
            hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
            h = hs[torch.arange(hs.size(0)), am.sum(1) - 1]
            outs.append(critic.value_head(h).float().cpu())
        return torch.cat(outs, 0)

    mrow = matrix.setdefault(cname, {})
    for v in VARIANTS:
        if v in mrow:
            continue
        P = recon(T[v])
        Pn = P / P.norm(dim=-1, keepdim=True) * ms
        mrow[v] = 1 - ((Pn - Gn) ** 2).mean().item() / den
        print(f"  {cname} x {v}: FVE={mrow[v]:.4f}", flush=True)
        json.dump(state, open(OUT, "w"))

    if cname == "v3rl" and state.get("saliency") is None:
        # leave-one-token-out saliency, 15 samples
        sal_out = []
        for i in range(15):
            text = T["orig"][i]
            ids = ctok.encode(text, add_special_tokens=False)[:80]
            base_P = recon([ctok.decode(ids)])
            bn = base_P / base_P.norm(dim=-1, keepdim=True) * ms
            base_err = ((bn - Gn[i]) ** 2).mean().item()
            cand = [ctok.decode(ids[:j] + ids[j+1:]) for j in range(len(ids))]
            P = recon(cand)
            Pn = P / P.norm(dim=-1, keepdim=True) * ms
            errs = ((Pn - Gn[i]) ** 2).mean(dim=1)
            deltas = (errs - base_err).tolist()
            toks = [ctok.decode([x]) for x in ids]
            top = sorted(zip(toks, deltas), key=lambda x: -x[1])[:8]
            sal_out.append({"i": i, "top_tokens": top})
            print(f"  saliency {i}: {[t for t, _ in top[:5]]}", flush=True)
        state["saliency"] = sal_out
        json.dump(state, open(OUT, "w"))
    del critic
    torch.cuda.empty_cache()

print("\n=== FVE matrix ===")
hdr = "variant            " + "".join(f"{c:>9s}" for c in CRITICS)
print(hdr)
for v in VARIANTS:
    print(f"{v:18s} " + "".join(f"{matrix[c][v]:>9.3f}" for c in CRITICS))
print("MATRIX_DONE", flush=True)
