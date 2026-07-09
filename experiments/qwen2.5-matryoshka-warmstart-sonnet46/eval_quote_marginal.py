"""Per-token marginal FVE, attributed to in-quote vs out-of-quote tokens.

For each saved explanation (100 held-out samples), reconstruct every token
prefix 1..len with the critic. The marginal contribution of token t in sample i
is (err_i(t-1) - err_i(t)) / den — err is the sample's normalized squared error
to its gold, den the set variance, so contributions sum to the sample's total
FVE share. Each token is classified in-quote if its character span (fast-
tokenizer offsets) intersects a "..." span (quote marks count as in-quote).

Env: AR_DIR, TEXTS_KEY (orig|ws_orig), EVAL, TRANSFORMS, OUT.
"""
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq
import torch

from nla_inference import NLACritic

AR_DIR = os.environ["AR_DIR"]
TEXTS_KEY = os.environ.get("TEXTS_KEY", "orig")
EVAL = os.environ.get("EVAL", "/workspace/out/av_eval_v3.parquet")
TRANSFORMS = os.environ.get("TRANSFORMS", "/workspace/code_invest_transforms.json")
OUT = os.environ["OUT"]
MAXTOK = 256
dev = "cuda"

SP = json.load(open(TRANSFORMS))
sel, texts = SP["sel"], SP["transforms"][TEXTS_KEY]
t = pq.read_table(EVAL)
G = torch.stack([torch.tensor(np.asarray(t.column("activation_vector")[i].as_py(), dtype=np.float32))
                 for i in sel]).float()

critic = NLACritic(AR_DIR, device=dev)
ms = critic.mse_scale
Gn = G / G.norm(dim=-1, keepdim=True) * ms
mu = Gn.mean(0)
den = ((Gn - mu) ** 2).mean().item()
ctok = critic.tokenizer
ctok.padding_side = "right"
if ctok.pad_token_id is None:
    ctok.pad_token = ctok.eos_token

QUOTE = re.compile(r'"[^"\n]{1,80}"')


@torch.no_grad()
def recon_err(texts_batch, gold_rows, bs=64):
    errs = []
    for i in range(0, len(texts_batch), bs):
        pr = [critic.template.format(explanation=e) for e in texts_batch[i:i+bs]]
        enc = ctok(pr, return_tensors="pt", add_special_tokens=True, padding=True)
        ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
        hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
        h = hs[torch.arange(hs.size(0)), am.sum(1) - 1]
        P = critic.value_head(h).float().cpu()
        Pn = P / P.norm(dim=-1, keepdim=True) * ms
        Grows = torch.stack([Gn[j] for j in gold_rows[i:i+bs]])
        errs.extend(((Pn - Grows) ** 2).mean(dim=1).tolist())
    return errs

out_rows = []
for i, text in enumerate(texts):
    enc = ctok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"][:MAXTOK]
    offs = enc["offset_mapping"][:MAXTOK]
    qspans = [(m.start(), m.end()) for m in QUOTE.finditer(text)]
    inq = [any(o0 < qe and o1 > qs for qs, qe in qspans) for o0, o1 in offs]
    prefixes = [ctok.decode(ids[:L]) for L in range(1, len(ids) + 1)]
    errs = recon_err(prefixes, [i] * len(prefixes))
    # err at L=0: predict-the-mean baseline error for this sample
    err0 = ((mu - Gn[i]) ** 2).mean().item()
    deltas = [(prev - cur) / den for prev, cur in zip([err0] + errs[:-1], errs)]
    toks = [ctok.decode([x]) for x in ids]
    out_rows.append({"i": i, "tokens": toks, "in_quote": inq, "delta_fve": deltas})
    if i % 10 == 0:
        print(f"  {i+1}/{len(texts)}", flush=True)

json.dump(out_rows, open(OUT, "w"))

allq = [d for r in out_rows for d, q in zip(r["delta_fve"], r["in_quote"]) if q]
alln = [d for r in out_rows for d, q in zip(r["delta_fve"], r["in_quote"]) if not q]
print(f"\n{TEXTS_KEY}: in-quote tokens {len(allq)} ({len(allq)/(len(allq)+len(alln)):.1%}), "
      f"mean ΔFVE {np.mean(allq):.5f} | out-of-quote {len(alln)}, mean ΔFVE {np.mean(alln):.5f}")
print(f"share of total positive-sum FVE from in-quote: "
      f"{sum(allq)/(sum(allq)+sum(alln)):.1%}")
print("QUOTE_MARGINAL_DONE", flush=True)
