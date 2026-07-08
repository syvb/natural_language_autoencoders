"""Stage 1: round-trip FVE vs truncation length (tokens AND lines).

Port of fve_truncation_sweep/sweep_fve.py: 100 held-out first-per-doc rows,
AV explanation at T=1 (dash prefill), then critic reconstruction of every
token-truncation (trimmed grid) and line-truncation. FVE = 1 - dirMSE/var,
both sides L2-normalized to mse_scale, mean baseline — identical definition.
Critic = the model's own co-trained rl_critic_step400.
Writes $WORK/suite/{token_fve,lines_fve}_$MODEL.csv + expls_$MODEL.json.
"""
import gc
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
import torch

from nla.models import NLACriticModel
from nla.schema import resolve_target_scale
from nla.utils import critic_predict
from suite_common import (CJK, CRITIC_DIR, MODEL, PARQUET, WORK, av_batch,
                          load_actor, load_base, load_tok_cfg, prompt_ids)

N = int(os.environ.get("N", "100"))
B = 16
os.makedirs(f"{WORK}/suite", exist_ok=True)

t = pq.read_table(PARQUET, columns=["prompt", "activation_vector", "doc_id"])
docs = t.column("doc_id").to_pylist()
seen = {}
for idx, d in enumerate(docs):
    if d not in seen:
        seen[d] = idx
first = list(seen.values())
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
prompts = [t.column("prompt")[i].as_py() for i in sel]
vecs = [np.asarray(t.column("activation_vector")[i].as_py(), np.float32) for i in sel]
print(f"[{MODEL}] selected {len(sel)} distinct-doc held-out samples", flush=True)

expl_path = f"{WORK}/suite/expls_{MODEL}.json"
if os.path.exists(expl_path):
    expls = json.load(open(expl_path))
    print("reusing cached explanations", flush=True)
else:
    vref = [None]
    actor, tok, cfg = load_actor(load_base(), vref)
    ids = prompt_ids(tok, cfg, prompts[0])
    torch.manual_seed(int(os.environ.get("NLA_GEN_SEED", "0")))
    expls, ncjk = [], 0
    for s in range(0, len(sel), B):
        for g in av_batch(actor, tok, vref, ids, vecs[s:s + B]):
            ncjk += bool(CJK.search(g))
            expls.append(g.strip())
        print(f"  AV gen {min(s+B, len(sel))}/{len(sel)} (cjk={ncjk})", flush=True)
    json.dump(expls, open(expl_path, "w"), indent=1)
    del actor
    gc.collect()
    torch.cuda.empty_cache()

tok, cfg = load_tok_cfg()
msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
critic = NLACriticModel.from_pretrained(
    CRITIC_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    device_map={"": 0}).eval()
template = cfg.critic_prompt_template
G = torch.tensor(np.stack(vecs), dtype=torch.float32)
pad = tok.eos_token_id


@torch.no_grad()
def reconstruct(texts, bs=24):
    outs = []
    idlists = [tok.encode(template.format(explanation=e), add_special_tokens=False)[:1024]
               for e in texts]
    for i in range(0, len(idlists), bs):
        chunk = idlists[i:i + bs]
        m = max(len(x) for x in chunk)
        bx = torch.full((len(chunk), m), pad, dtype=torch.long, device="cuda")
        attn = torch.zeros((len(chunk), m), dtype=torch.long, device="cuda")
        for r, q in enumerate(chunk):
            bx[r, :len(q)] = torch.tensor(q, dtype=torch.long)
            attn[r, :len(q)] = 1
        outs.append(critic_predict(critic, bx, attn, msf).float().cpu())
    return torch.cat(outs, 0)


def fve(P, Gm):
    gn = Gm / Gm.norm(dim=-1, keepdim=True) * msf
    pn = P / P.norm(dim=-1, keepdim=True) * msf
    mu = gn.mean(0)
    return 1 - ((pn - gn) ** 2).mean().item() / ((gn - mu) ** 2).mean().item()


tok_ids = [tok.encode(e, add_special_tokens=False) for e in expls]
lens = [len(x) for x in tok_ids]
Lmax = max(lens)
grid = sorted(set(list(range(1, 21)) + list(range(22, 61, 2)) +
                  list(range(65, 151, 5)) + list(range(160, Lmax + 1, 10)) + [Lmax]))
grid = [L for L in grid if L <= Lmax]
print(f"TOKEN sweep: Lmax={Lmax} median={int(np.median(lens))} grid={len(grid)} pts", flush=True)
tok_rows = []
for L in grid:
    P = reconstruct([tok.decode(x[:L]) for x in tok_ids])
    f = fve(P, G)
    tok_rows.append((L, f, sum(1 for x in lens if x <= L)))
    print(f"  L={L:>3} FVE={f:.4f}", flush=True)

lines_per = [[p for p in re.split(r"\n+", e.strip()) if p.strip()] for e in expls]
klens = [len(x) for x in lines_per]
line_rows = []
for K in range(1, max(klens) + 1):
    P = reconstruct(["\n".join(lp[:K]) for lp in lines_per])
    f = fve(P, G)
    line_rows.append((K, f, sum(1 for x in klens if x <= K)))
    print(f"  K={K:>2} FVE={f:.4f}", flush=True)

fve_full = fve(reconstruct(expls), G)
with open(f"{WORK}/suite/token_fve_{MODEL}.csv", "w") as fh:
    fh.write("length_tokens,fve,n_samples_at_full\n")
    for L, f, n in tok_rows:
        fh.write(f"{L},{f:.6f},{n}\n")
with open(f"{WORK}/suite/lines_fve_{MODEL}.csv", "w") as fh:
    fh.write("length_lines,fve,n_samples_at_full\n")
    for K, f, n in line_rows:
        fh.write(f"{K},{f:.6f},{n}\n")
print(f"full-len FVE={fve_full:.4f}")
print("FVE_DONE", flush=True)
