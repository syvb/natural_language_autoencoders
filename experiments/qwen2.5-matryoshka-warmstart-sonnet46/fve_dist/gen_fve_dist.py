"""Per-ROLLOUT round-trip FVE on held-out eval examples (distribution, not aggregate).

For N distinct-doc held-out samples x NROLL sampled rollouts each (T=1 default):
  AV (transformers+injection): gold activation v -> explanation text
  AR (NLACritic):              text -> v_hat, at FULL length and a 10-content-token prefix
Per-rollout FVE_i = 1 - mean((pn_i - gn_i)^2) / denom, where pn/gn are L2-normalized
to mse_scale and denom = mean((gn_j - mu)^2) over the N unique golds (population
variance, raw-mean baseline). This makes mean_i(FVE_i) equal the aggregate FVE of
eval_round_trip_fve.py / sweep_fve.py, so the histogram decomposes the headline number.

Sample selection (SELECT=doc, default) is identical to sweep_fve.py (first
occurrence per doc, stride), computed over the FULL N before sharding, so shards
partition one fixed sample set and the denominator is shard-independent.
SELECT=row strides over ALL eval rows instead (doc_ids repeat; use when N exceeds
the distinct-doc count).

Env: AV_DIR, AR_DIR, EVAL, OUT (json path), N (default 250), NROLL (default 2),
     SELECT (doc|row, default doc), SHARD/NSHARDS (default 0/1),
     NLA_GEN_TEMP (default 1), NLA_GEN_SEED (default 0).
"""
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import INJECT_PLACEHOLDER, extract_explanation_open, normalize_activation
from nla_inference import NLACritic

AV_DIR = os.environ["AV_DIR"]
AR_DIR = os.environ["AR_DIR"]
EVAL = os.environ["EVAL"]
OUT = os.environ["OUT"]
N = int(os.environ.get("N", "250"))
NROLL = int(os.environ.get("NROLL", "2"))
SELECT = os.environ.get("SELECT", "doc")
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))
_T = float(os.environ.get("NLA_GEN_TEMP", "1"))
GEN_KW = (dict(do_sample=True, temperature=_T, top_p=1.0, top_k=0)
          if _T > 0 else dict(do_sample=False))
_SEED = int(os.environ.get("NLA_GEN_SEED", "0"))
PREFIX = 10
BG = 16
dev = "cuda"

meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml"))
T = meta["tokens"]; inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]
right = T["injection_right_neighbor_id"]; inj_char = T["injection_char"]
inj_scale = meta["extraction"]["injection_scale"]

tok = AutoTokenizer.from_pretrained(AV_DIR); tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
av = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16).to(dev).eval()
emb = av.get_input_embeddings()

# --- sample selection (stride over candidates); shard AFTER selection ---
t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
if SELECT == "doc":  # same N distinct-doc samples as sweep_fve.py
    seen = {}
    for idx, d in enumerate(docs):
        if d not in seen: seen[d] = idx
    first = list(seen.values())
else:  # "row": every eval row is a candidate
    first = list(range(len(docs)))
assert N <= len(first), (N, len(first))
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
col = t.column
all_vecs = [np.asarray(col("activation_vector")[i].as_py(), dtype=np.float32) for i in sel]
my_idx = list(range(N))[SHARD::NSHARDS]
prompts = [col("prompt")[sel[i]].as_py() for i in my_idx]
my_docs = [docs[sel[i]] for i in my_idx]
vecs = [all_vecs[i] for i in my_idx]
print(f"N={N} select={SELECT} shard {SHARD}/{NSHARDS}: {len(my_idx)} samples x {NROLL} rollouts, T={_T}", flush=True)


@torch.no_grad()
def av_generate(bp, bv):
    seqs = []
    for p in bp:
        msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in p]
        seqs.append(tok.apply_chat_template(msgs, add_generation_prompt=True))
    m = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), m), pad, dtype=np.int64); att = np.zeros((len(seqs), m), dtype=np.int64)
    for k, s in enumerate(seqs):
        inp[k, m - len(s):] = s; att[k, m - len(s):] = 1
    inp = torch.tensor(inp, device=dev); att = torch.tensor(att, device=dev)
    e = emb(inp)
    V = torch.stack([normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), inj_scale)[0] for v in bv])
    out = av.generate(inputs_embeds=inject_at_marked_positions(inp, e, V, inj_id, left, right),
                      attention_mask=att, max_new_tokens=256, pad_token_id=pad, **GEN_KW)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")
rows = []
for rep in range(NROLL):
    torch.manual_seed(_SEED * 100003 + rep * 977 + SHARD * 13)
    for s in range(0, len(my_idx), BG):
        for j, txt in enumerate(av_generate(prompts[s:s+BG], vecs[s:s+BG])):
            e = extract_explanation_open(txt) or txt
            rows.append(dict(i=my_idx[s + j], doc_id=my_docs[s + j], rep=rep,
                             cjk=int(bool(CJK.search(txt))), text=e))
        print(f"  rep {rep}: AV gen {min(s+BG,len(my_idx))}/{len(my_idx)}", flush=True)

del av, emb; torch.cuda.empty_cache()
critic = NLACritic(AR_DIR, device=dev)
ms = critic.mse_scale
ctok = critic.tokenizer
ctok.padding_side = "right"
if ctok.pad_token_id is None: ctok.pad_token = ctok.eos_token


@torch.no_grad()
def reconstruct_batch(explanations, bs=32):
    outs = []
    for i in range(0, len(explanations), bs):
        chunk = explanations[i:i+bs]
        pr = [critic.template.format(explanation=e) for e in chunk]
        enc = ctok(pr, return_tensors="pt", add_special_tokens=True, padding=True)
        ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
        hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
        last = am.sum(1) - 1
        h = hs[torch.arange(hs.size(0)), last]
        outs.append(critic.value_head(h).float().cpu())
    return torch.cat(outs, 0)


def nrm(X):
    return X / X.norm(dim=-1, keepdim=True) * ms


# denominator over the FULL N golds (shard-independent, matches sweep_fve's fve())
Gall = nrm(torch.stack([torch.tensor(v) for v in all_vecs]).float())
mu = Gall.mean(0)
denom = ((Gall - mu) ** 2).mean().item()

G = torch.stack([Gall[r["i"]] for r in rows])
texts = [r["text"] for r in rows]
tok_ids = [ctok.encode(e, add_special_tokens=False) for e in texts]
P_full = nrm(reconstruct_batch(texts))
P_p10 = nrm(reconstruct_batch([ctok.decode(ids[:PREFIX]) for ids in tok_ids]))
for k, r in enumerate(rows):
    r["ntok"] = len(tok_ids[k])
    for tag, P in (("full", P_full), ("p10", P_p10)):
        r[f"err2_{tag}"] = ((P[k] - G[k]) ** 2).mean().item()
        r[f"cos_{tag}"] = torch.nn.functional.cosine_similarity(P[k], G[k], dim=0).item()

fves = [1 - r["err2_full"] / denom for r in rows]
print(f"shard aggregate: full FVE={np.mean(fves):.4f}  "
      f"p10 FVE={np.mean([1 - r['err2_p10'] / denom for r in rows]):.4f}  "
      f"cjk={sum(r['cjk'] for r in rows)}/{len(rows)}", flush=True)
json.dump(dict(denom=denom, mse_scale=ms, n=N, nroll=NROLL, temp=_T, prefix=PREFIX,
               select=SELECT, av_dir=AV_DIR, eval_parquet=os.path.basename(EVAL), rows=rows),
          open(OUT, "w"))
print("FVEDIST_DONE", flush=True)
