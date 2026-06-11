"""Gate D extraction: attention-block writes at L20 + mechanical evidence.

Per position stores (raw fp32, norm="none" per repo invariant):
  attn_out.npy  — o_proj output at p (the block's residual write; the target)
  v_pre.npy     — hidden_states[20] at p (residual state ENTERING layer 20;
                  for the D0.2 content-confound ridge and the arm-C control)
  mlp_out.npy   — layer-20 MLP write at p (free descriptive arm)
  attn_evidence.jsonl — per position: total write norm, dominant heads
                  (covering 90% of contribution norm, max 8) with per-head
                  fraction and top-5 attended sources (position, weight,
                  token, ±3-token context), plus logit-lens top/bottom 10
                  of the write (soft evidence).
  meta_d.json, doc_tokens.jsonl — as in gate C.

Layer indexing: hidden_states[20] is the INPUT to decoder layer index 20;
the attention sub-layer of model.model.layers[20] reads it and writes
attn_out into the stream. attentions[20] are that block's weights (needs
attn_implementation="eager").

Sanity asserted per doc: sum of per-head writes == o_proj output (the GQA
o_proj has no bias), cos > 0.999.
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
LAYER = 20
N_DOCS = 2500
POS_PER_DOC = 6
MIN_POSITION = 50
MAX_TOKENS = 512
SEED = 2  # fresh corpus slice (gate A used 0, gate C used 1)
HOLDOUT_FRAC = 0.2
TOPK_SRC = 5
MAX_HEADS = 8
HEAD_COVER = 0.90

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
    attn_implementation="eager", device_map="cuda").eval()
special_ids = set(tok.all_special_ids)
rng = np.random.default_rng(SEED)
torch.set_grad_enabled(False)  # per-head writes use W_O directly

layer = model.model.layers[LAYER]
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
W_O = layer.self_attn.o_proj.weight  # [d, n_heads*head_dim], no bias in Qwen2

cap = {}
h1 = layer.self_attn.o_proj.register_forward_pre_hook(
    lambda m, a: cap.__setitem__("oin", a[0].detach()))
h2 = layer.self_attn.o_proj.register_forward_hook(
    lambda m, i, o: cap.__setitem__("attn", o.detach()))
h3 = layer.mlp.register_forward_hook(
    lambda m, i, o: cap.__setitem__("mlp", o.detach()))

ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train", streaming=True)
out_attn, out_vpre, out_mlp = [], [], []
meta = []
n_done = 0
fev = open(WORK / "attn_evidence.jsonl", "w")
ftok = open(WORK / "doc_tokens.jsonl", "w")

for doc_idx, doc in enumerate(ds):
    ids = tok(doc["text"], add_special_tokens=False,
              truncation=True, max_length=MAX_TOKENS)["input_ids"]
    candidates = [i for i, t in enumerate(ids)
                  if i >= MIN_POSITION and t not in special_ids]
    if len(candidates) < POS_PER_DOC + 10:
        continue
    positions = sorted(rng.choice(candidates, POS_PER_DOC, replace=False).tolist())
    split = "holdout" if rng.random() < HOLDOUT_FRAC else "train"
    x = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(x, output_hidden_states=True, output_attentions=True)
    attw = out.attentions[LAYER][0].float()       # [n_heads, seq, seq]
    vpre_seq = out.hidden_states[LAYER][0]        # input to layer 20
    attn_seq, mlp_seq, oin_seq = cap["attn"][0], cap["mlp"][0], cap["oin"][0]

    for pos in positions:
        a_p = attn_seq[pos].float()
        oin = oin_seq[pos].float().view(n_heads, head_dim)
        writes = torch.stack([
            W_O.float()[:, h * head_dim:(h + 1) * head_dim] @ oin[h]
            for h in range(n_heads)])                       # [n_heads, d]
        recon = writes.sum(0)
        cos = torch.nn.functional.cosine_similarity(recon, a_p, dim=0).item()
        assert cos > 0.999, f"head-write decomposition broke: cos={cos:.4f}"
        contribs = writes.norm(dim=-1)
        fracs = (contribs / contribs.sum().clamp_min(1e-9)).cpu().numpy()
        order = np.argsort(-fracs)
        heads_ev = []
        cum = 0.0
        for h in order[:MAX_HEADS]:
            if cum >= HEAD_COVER:
                break
            cum += float(fracs[h])
            w = attw[h, pos]
            topv, topj = w.topk(min(TOPK_SRC, pos + 1))
            srcs = [{"j": int(j), "w": round(float(v), 4),
                     "tok": tok.decode([ids[int(j)]]),
                     "ctx": tok.decode(ids[max(0, int(j) - 3):int(j) + 4])}
                    for v, j in zip(topv, topj)]
            heads_ev.append({"h": int(h), "frac": round(float(fracs[h]), 4),
                             "src": srcs})
        with torch.inference_mode():
            lens = model.lm_head(model.model.norm(a_p.to(torch.bfloat16))).float()
        lt, li = lens.topk(10)
        lb, lbi = (-lens).topk(10)
        fev.write(json.dumps({
            "i": len(meta), "total_norm": round(float(a_p.norm()), 3),
            "vpre_norm": round(float(vpre_seq[pos].float().norm()), 3),
            "heads": heads_ev,
            "lens_top": [tok.decode([int(t)]) for t in li],
            "lens_bot": [tok.decode([int(t)]) for t in lbi]}) + "\n")
        out_attn.append(a_p.cpu().numpy())
        out_vpre.append(vpre_seq[pos].float().cpu().numpy())
        out_mlp.append(mlp_seq[pos].float().cpu().numpy())
        meta.append({"doc_idx": doc_idx, "pos": pos, "split": split,
                     "token_at_pos": tok.decode([ids[pos]]),
                     "context_tail": tok.decode(ids[max(0, pos - 60):pos + 1],
                                                skip_special_tokens=True)})
    ftok.write(json.dumps({"doc_idx": doc_idx, "ids": ids}) + "\n")
    n_done += 1
    if n_done % 100 == 0:
        print(f"[extract_d] {n_done}/{N_DOCS} docs", flush=True)
    if n_done >= N_DOCS:
        break

fev.close(); ftok.close()
np.save(WORK / "attn_out.npy", np.stack(out_attn))
np.save(WORK / "v_pre.npy", np.stack(out_vpre))
np.save(WORK / "mlp_out.npy", np.stack(out_mlp))
(WORK / "meta_d.json").write_text(json.dumps(meta))
n_hold = sum(m["split"] == "holdout" for m in meta)
an = np.linalg.norm(np.stack(out_attn), axis=1)
print(f"[extract_d] saved {len(meta)} positions ({n_hold} holdout)")
print(f"[extract_d] attn_out norms: p10 {np.percentile(an,10):.2f} "
      f"median {np.median(an):.2f} p90 {np.percentile(an,90):.2f}")
