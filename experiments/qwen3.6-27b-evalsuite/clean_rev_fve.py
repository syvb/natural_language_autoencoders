"""Reversed-order FVE truncation sweep (clean held-out, both 27B NLAs).

Question: is the standard NLA's late-rising truncation curve an artifact of it
BACKLOADING information? Reverse the order of its explanation units before
token-truncating — if the tail carried the load, the reversed curve should rise
early like the matryoshka's.

Reuses the exact clean-held-out protocol of final_runs.py (the run that
produced results/clean_std_token_fve.csv): Ultra-FineWeb-en part-0001 rows
300000+, seed-42 doc/position selection, fresh L42 extraction on the pristine
base, T=1 generation (mat: no prefill; std: `<explanation>\n` prefill, cut at
`</explanation>`), own-critic FVE with both sides L2-normalized to mse_scale.

Sweeps (token truncation, same grid formula as the original), std only —
the matryoshka comparison curve is the checked-in clean_token_fve.csv:
  std : original line order | reversed line order | reversed sentence order
The original-order sweep doubles as a replication check against the
checked-in clean_std_token_fve.csv.

Box layout (see rev_setup.sh): /workspace/{base,ckpt_mat,ckpt_std,data},
outputs to /workspace/final/. Artifact-gated; safe to rerun.
"""
import gc
import json
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import resolve_target_scale
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

W = "/workspace"
F = f"{W}/final"
os.makedirs(F, exist_ok=True)
MAT = f"{W}/ckpt_mat"
STD = f"{W}/ckpt_std"
PARQUET = f"{W}/data/av_eval.parquet"
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")
N = 100
SEED = 42
POS_MIN, POS_MAX = 50, 480

tok = AutoTokenizer.from_pretrained(f"{MAT}/warmstart_av_lora")
cfg = load_nla_config(PARQUET, tok)
msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        f"{W}/base", torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()


# ---------- stage 0: clean activations (identical to final_runs.py) ----------
ACTS = f"{F}/clean_acts.npy"
if not os.path.exists(ACTS):
    shard = hf_hub_download("openbmb/Ultra-FineWeb",
                            "data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet",
                            repo_type="dataset")
    pf = pq.ParquetFile(shard)
    texts, cum = [], 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if cum + n <= 300000:
            cum += n
            continue
        t = pf.read_row_group(rg, columns=["content"]).column("content").to_pylist()
        for c in t[max(0, 300000 - cum):]:
            texts.append(c)
            if len(texts) >= 260:
                break
        cum += n
        if len(texts) >= 260:
            break
    rng = random.Random(SEED)
    picked = []
    for c in texts:
        ids = tok.encode(c, add_special_tokens=True)
        if len(ids) < POS_MIN + 20:
            continue
        picked.append((ids, rng.randint(POS_MIN, min(len(ids) - 1, POS_MAX))))
        if len(picked) >= N:
            break
    assert len(picked) >= N
    base = load_base()
    grabbed = {}
    h = resolve_decoder_layers(base)[42].register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    acts = np.empty((N, cfg.d_model), np.float32)
    for k, (ids, p) in enumerate(picked):
        with torch.no_grad():
            base(input_ids=torch.tensor([ids[:p + 1]], device="cuda"))
        acts[k] = grabbed["h"][0, p].float().cpu().numpy()
    h.remove()
    np.save(ACTS, acts)
    del base
    gc.collect(); torch.cuda.empty_cache()
    print("STAGE0_DONE", flush=True)
acts = np.load(ACTS)

pmsgs = pq.read_table(PARQUET, columns=["prompt"]).column("prompt")[0].as_py()


def prompt_ids(extra=""):
    ptxt = build_prompt_text(pmsgs, cfg.injection_char, tok)
    assert ptxt.endswith(THINK_OPEN)
    return tok.encode(ptxt[: -len(THINK_OPEN)] + PRECLOSED + extra, add_special_tokens=False)


@torch.no_grad()
def gen_batch(actor, vref, ids, vecs, max_new=256):
    pt = torch.tensor([ids], dtype=torch.long, device="cuda").repeat(len(vecs), 1)
    vref[0] = torch.tensor(np.stack(vecs), dtype=torch.float32).cuda()
    try:
        out = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                             max_new_tokens=max_new, do_sample=True, temperature=1.0,
                             top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    finally:
        vref[0] = None
    return [tok.decode(o[pt.shape[1]:], skip_special_tokens=True) for o in out]


# ---------- stage 1: standard explanations (tag protocol) ----------
SE = f"{F}/clean_expls_std.json"
if not os.path.exists(SE):
    base = load_base()
    merged = PeftModel.from_pretrained(base, f"{STD}/av_sft_lora").merge_and_unload()
    actor = PeftModel.from_pretrained(merged, f"{STD}/av_rl_lora_step400").eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    ids = prompt_ids("<explanation>\n")
    torch.manual_seed(0)
    expls, ncjk = [], 0
    B = 32
    for s in range(0, N, B):
        for g in gen_batch(actor, vref, ids, [acts[i] for i in range(s, min(s + B, N))], max_new=300):
            ncjk += bool(CJK.search(g))
            expls.append(g.split("</explanation>")[0].strip())
        print(f"  std expl {len(expls)}/{N} (cjk={ncjk})", flush=True)
    json.dump(expls, open(SE, "w"))
    del actor, merged, base
    gc.collect(); torch.cuda.empty_cache()
print("STAGE1_DONE", flush=True)


# ---------- reorderings ----------
def lines_of(e):
    return [ln.strip() for ln in re.split(r"\n+", e.strip()) if ln.strip()]


def sent_units(line):
    """Quote-parity-aware sentence slicing (same as std_loo_precompute.py)."""
    bounds = []
    for m in re.finditer(r'[.!?]["”\')\]]*\s+(?=[A-Z"“(\d])', line):
        prefix = line[: m.end()]
        if (prefix.count('"') + prefix.count('“') + prefix.count('”')) % 2 == 0:
            bounds.append(m.end())
    units, prev = [], 0
    for b in bounds:
        units.append(line[prev:b]); prev = b
    units.append(line[prev:])
    return [u.strip() for u in units if u.strip()]


def rev_lines(e):
    return "\n".join(reversed(lines_of(e)))


def rev_sents(e):
    units = [u for ln in lines_of(e) for u in sent_units(ln)]
    return " ".join(reversed(units))


# ---------- own-critic token sweeps ----------
G = torch.tensor(acts, dtype=torch.float32)
gn = G / G.norm(dim=-1, keepdim=True) * msf
var = float(((gn - gn.mean(0)) ** 2).mean())
pad = tok.eos_token_id
summary_path = f"{F}/rev_summary.json"
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

JOBS = [("std", f"{STD}/rl_critic_step400", SE,
         [("orig", lambda e: e), ("revlines", rev_lines), ("revsents", rev_sents)])]

for model, cdir, epath, orders in JOBS:
    todo = [(oname, fn) for oname, fn in orders
            if not os.path.exists(f"{F}/rev_token_fve_{model}_{oname}.csv")]
    if not todo:
        continue
    expls = json.load(open(epath))
    critic = NLACriticModel.from_pretrained(cdir, torch_dtype=torch.bfloat16,
                                            attn_implementation="sdpa",
                                            device_map={"": 0}).eval()

    @torch.no_grad()
    def recon(texts, bs=64):
        outs = []
        idl = [tok.encode(cfg.critic_prompt_template.format(explanation=e),
                          add_special_tokens=False)[:1024] for e in texts]
        for i in range(0, len(idl), bs):
            ch = idl[i:i + bs]
            m = max(len(x) for x in ch)
            bx = torch.full((len(ch), m), pad, dtype=torch.long, device="cuda")
            at = torch.zeros((len(ch), m), dtype=torch.long, device="cuda")
            for r, q in enumerate(ch):
                bx[r, :len(q)] = torch.tensor(q)
                at[r, :len(q)] = 1
            outs.append(critic_predict(critic, bx, at, msf).float().cpu())
        return torch.cat(outs, 0)

    def fve_of(P):
        pn = P / P.norm(dim=-1, keepdim=True) * msf
        return 1 - ((pn - gn) ** 2).mean().item() / var

    for oname, fn in todo:
        texts = [fn(e) for e in expls]
        tok_ids = [tok.encode(t, add_special_tokens=False) for t in texts]
        Lmax = max(len(x) for x in tok_ids)
        grid = sorted(set(list(range(1, 21)) + list(range(22, 61, 2))
                          + list(range(65, 151, 5))
                          + list(range(160, Lmax + 1, 10)) + [Lmax]))
        rows = []
        for L in [g for g in grid if g <= Lmax]:
            f = fve_of(recon([tok.decode(x[:L]) for x in tok_ids]))
            rows.append((L, f, sum(1 for x in tok_ids if len(x) <= L)))
            print(f"  {model}/{oname} L={L:>3} FVE={f:.4f}", flush=True)
        with open(f"{F}/rev_token_fve_{model}_{oname}.csv", "w") as fh:
            fh.write("length_tokens,fve,n_samples_at_full\n")
            for L, f, n in rows:
                fh.write(f"{L},{f:.6f},{n}\n")
        summary[f"{model}|{oname}|full"] = fve_of(recon(texts))
        json.dump(summary, open(summary_path, "w"), indent=1)
        print(f"{model}/{oname}: full-len FVE={summary[f'{model}|{oname}|full']:.4f}", flush=True)
    del critic
    gc.collect(); torch.cuda.empty_cache()

print("REV_ALL_DONE", flush=True)
