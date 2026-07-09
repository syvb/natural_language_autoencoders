"""Critic-only re-reconstruction of saved fve_dist rollouts at a grid of prefix budgets.

Reads the fvedist_<model>_s*.json files produced by gen_fve_dist.py (which store
the rollout TEXTS), reconstructs each rollout at every prefix length in PREFIXES
(content tokens of the critic tokenizer; texts shorter than k are used whole,
matching sweep_fve.py), and writes per-rollout err2/cos per budget. No AV /
generation involved, so this reruns cheaply for any budget question.

Gold activations are re-read from the eval parquet with the same selection logic
as gen_fve_dist.py (SELECT mode taken from the input JSONs: first occurrence per
doc, or all rows, then stride over N); rows carry the selection index `i`. The
denominator is recomputed over all N golds and asserted to match the one stored
in the input JSONs.

Env: AR_DIR, EVAL, IN_GLOB, OUT, PREFIXES (default "1,2,3,5,7,10,15,20,25,30,40,50,60,80,100,120,160,200").
"""
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from nla_inference import NLACritic

AR_DIR = os.environ["AR_DIR"]
EVAL = os.environ["EVAL"]
IN_GLOB = os.environ["IN_GLOB"]
OUT = os.environ["OUT"]
PREFIXES = [int(x) for x in os.environ.get(
    "PREFIXES", "1,2,3,5,7,10,15,20,25,30,40,50,60,80,100,120,160,200").split(",")]
dev = "cuda"

rows, denom0 = [], None
for p in sorted(glob.glob(IN_GLOB)):
    d = json.load(open(p))
    rows += d["rows"]; denom0 = d["denom"]; N = d["n"]
    SELECT = d.get("select", "doc")
print(f"{len(rows)} rollouts from {IN_GLOB}, N={N}, select={SELECT}", flush=True)

t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
if SELECT == "doc":
    seen = {}
    for idx, dd in enumerate(docs):
        if dd not in seen: seen[dd] = idx
    first = list(seen.values())
else:
    first = list(range(len(docs)))
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
all_vecs = [np.asarray(t.column("activation_vector")[i].as_py(), dtype=np.float32) for i in sel]
for r in rows[:1] + rows[-1:]:
    assert docs[sel[r["i"]]] == r["doc_id"], (r["i"], r["doc_id"])

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


Gall = nrm(torch.stack([torch.tensor(v) for v in all_vecs]).float())
mu = Gall.mean(0)
denom = ((Gall - mu) ** 2).mean().item()
assert abs(denom - denom0) < 1e-6, (denom, denom0)

G = torch.stack([Gall[r["i"]] for r in rows])
tok_ids = [ctok.encode(r["text"], add_special_tokens=False) for r in rows]
out_rows = [dict(i=r["i"], doc_id=r["doc_id"], rep=r["rep"], cjk=r["cjk"],
                 ntok=len(ids)) for r, ids in zip(rows, tok_ids)]
for k in PREFIXES:
    P = nrm(reconstruct_batch([ctok.decode(ids[:k]) for ids in tok_ids]))
    e2 = ((P - G) ** 2).mean(-1)
    cs = torch.nn.functional.cosine_similarity(P, G, dim=-1)
    for j, r in enumerate(out_rows):
        r[f"err2_k{k}"] = e2[j].item(); r[f"cos_k{k}"] = cs[j].item()
    print(f"  k={k:>3}  aggregate FVE={1 - e2.mean().item() / denom:+.4f}", flush=True)

json.dump(dict(denom=denom, mse_scale=ms, n=N, prefixes=PREFIXES, select=SELECT,
               ar_dir=AR_DIR, eval_parquet=os.path.basename(EVAL), rows=out_rows),
          open(OUT, "w"))
print("RECON_BUDGET_DONE", flush=True)
