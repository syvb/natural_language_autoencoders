"""Suffix-RL samples + dense per-token FVE sweep (both directions), one box pass.

Part 1 — example outputs: N_GEN held-out activations (doc-disjoint av_eval_v3),
each verbalized by the ws AV and the suffix-RL iter_50 AV (T=1, 120-token cap,
matching training). First N_MD go into suffix_rl_results/example_outputs.md
with per-sample cos (full / prefix@10 / suffix@10) through each pair's own critic.

Part 2 — dense sweep: for every generation, FVE at EVERY cut L=1..120 for both
sides (prefix ids[:L], suffix ids[-L:]; L past the text length repeats the full
text, so every L averages all samples — marginal contribution is then 0).
Writes sweep_dense.csv: arm,side,L,fve — the input to plot_marginal_fve.py.

Models are loaded ONE at a time (AVs then critics) to stay far under 80GB.
Run:  cd /root && PYTHONPATH=/workspace/nla python .../gen_samples_and_sweep.py
Env:  N_GEN (default 60), N_MD (15), OUT_DIR (/workspace/out)
"""
import gc
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import INJECT_PLACEHOLDER, extract_explanation_open, normalize_activation
from nla_inference import NLACritic

EVAL = os.environ.get("EVAL", "/workspace/data/v3/av_eval_v3.parquet")
OUT = os.environ.get("OUT_DIR", "/workspace/out")
N_GEN = int(os.environ.get("N_GEN", "60"))
N_MD = int(os.environ.get("N_MD", "15"))
CAP = 120
ARMS = {  # name -> (av_dir, ar_dir)
    "ws":   ("/workspace/m/ws_actor", "/workspace/m/ws_critic"),
    "it50": ("/workspace/m/sfx50/iter_0000050/av", "/workspace/m/sfx50/iter_0000050/ar"),
}
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

# ---- held-out sample selection (same round-robin-by-doc as eval_round_trip_fve) ----
t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
seen: dict = {}
for i, d in enumerate(docs):
    seen.setdefault(d, []).append(i)
order, k = [], 0
docl = list(seen.values())
while len(order) < N_GEN and any(k < len(v) for v in docl):
    for v in docl:
        if k < len(v):
            order.append(v[k])
        if len(order) >= N_GEN:
            break
    k += 1
order = order[:N_GEN]
col = t.column
prompts = [col("prompt")[i].as_py() for i in order]
vecs = [np.asarray(col("activation_vector")[i].as_py(), dtype=np.float32) for i in order]
ctxs = [col("detokenized_text_truncated")[i].as_py() if "detokenized_text_truncated" in t.column_names else "" for i in order]
print(f"{len(order)} samples from {len(set(docs[i] for i in order))} docs", flush=True)

# ---- part 1: generations, one AV at a time ----
gens: dict[str, list[str]] = {}
for arm, (av_dir, _) in ARMS.items():
    meta = yaml.safe_load(open(f"{av_dir}/nla_meta.yaml"))
    T = meta["tokens"]
    inj_id, left, right, inj_char = (T["injection_token_id"], T["injection_left_neighbor_id"],
                                     T["injection_right_neighbor_id"], T["injection_char"])
    inj_scale = meta["extraction"]["injection_scale"]
    tok = AutoTokenizer.from_pretrained(av_dir)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(av_dir, dtype=torch.bfloat16).to(dev).eval()
    emb = av.get_input_embeddings()
    outs = []
    B = 16
    with torch.no_grad():
        for s in range(0, len(order), B):
            bp, bv = prompts[s:s + B], vecs[s:s + B]
            seqs = []
            for p in bp:
                msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in p]
                seqs.append(tok.apply_chat_template(msgs, add_generation_prompt=True))
            m_len = max(len(x) for x in seqs)
            pad = tok.pad_token_id
            inp = np.full((len(seqs), m_len), pad, dtype=np.int64)
            att = np.zeros((len(seqs), m_len), dtype=np.int64)
            for j, sq in enumerate(seqs):
                inp[j, m_len - len(sq):] = sq
                att[j, m_len - len(sq):] = 1
            inp_t = torch.tensor(inp, device=dev); att_t = torch.tensor(att, device=dev)
            e = emb(inp_t)
            V = torch.stack([normalize_activation(torch.tensor(v).view(1, -1), inj_scale)[0] for v in bv])
            e2 = inject_at_marked_positions(inp_t, e, V, inj_id, left, right)
            out = av.generate(inputs_embeds=e2, attention_mask=att_t, max_new_tokens=CAP,
                              do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                              pad_token_id=pad)
            outs += [extract_explanation_open(tok.decode(o, skip_special_tokens=True)) or
                     tok.decode(o, skip_special_tokens=True) for o in out]
            print(f"  {arm}: {min(s + B, len(order))}/{len(order)}", flush=True)
    gens[arm] = outs
    del av, emb
    gc.collect(); torch.cuda.empty_cache()
# AV tokenizer for the cuts (identical across arms — same Qwen family)
cut_tok = AutoTokenizer.from_pretrained(ARMS["ws"][0])


def cut(text, L, side):
    ids = cut_tok.encode(text, add_special_tokens=False)
    ids = ids[:L] if side == "prefix" else ids[-L:]
    return cut_tok.decode(ids)


# ---- part 2: dense both-sides sweep + per-sample stats, one critic at a time ----
G = torch.stack([torch.tensor(v) for v in vecs]).float()
rows = ["arm,side,L,fve"]
persample: dict[str, dict] = {a: {} for a in ARMS}
for arm, (_, ar_dir) in ARMS.items():
    critic = NLACritic(ar_dir, device=dev)
    ms = critic.mse_scale
    gn = G / G.norm(dim=-1, keepdim=True) * ms
    mu = gn.mean(0)
    den = ((gn - mu) ** 2).mean().item()

    def fve_of(preds):  # preds [N, d]
        pn = preds / preds.norm(dim=-1, keepdim=True) * ms
        return 1 - ((pn - gn) ** 2).mean().item() / den

    def cos_row(pred, gold):
        return torch.nn.functional.cosine_similarity(pred, gold, dim=0).item()

    full_preds = torch.stack([critic.reconstruct(g) for g in gens[arm]]).float()
    persample[arm]["full_cos"] = [cos_row(full_preds[i], G[i]) for i in range(len(order))]
    for tag, side in (("p10", "prefix"), ("s10", "suffix")):
        pr = torch.stack([critic.reconstruct(cut(g, 10, side)) for g in gens[arm]]).float()
        persample[arm][tag + "_cos"] = [cos_row(pr[i], G[i]) for i in range(len(order))]
    for side in ("prefix", "suffix"):
        for L in range(1, CAP + 1):
            texts = [cut(g, L, side) for g in gens[arm]]
            preds = []
            for s in range(0, len(texts), 64):
                preds.append(critic.reconstruct_batch(texts[s:s + 64]).float())
            rows.append(f"{arm},{side},{L},{fve_of(torch.cat(preds)):.6f}")
        print(f"  {arm}/{side} dense done", flush=True)
    del critic
    gc.collect(); torch.cuda.empty_cache()
open(f"{OUT}/sweep_dense.csv", "w").write("\n".join(rows) + "\n")

# ---- example_outputs.md ----
md = ["# Left-matryoshka NLA — example outputs\n",
      f"\n{N_MD} held-out activations (doc-disjoint `av_eval_v3`), verbalized by the"
      " **v3 warm-start AV** and the **suffix-RL iter_50 AV** (T=1, 120-token cap,"
      " as in training). Per-sample cosine through each pair's own critic:"
      " full text / first-10-tokens / last-10-tokens.\n"]
for i in range(min(N_MD, len(order))):
    md.append(f"\n---\n\n## Sample {i + 1}\n")
    if ctxs[i]:
        c = ctxs[i]
        md.append(f"\n**Context (last 300 chars):** `…{c[-300:]}`\n")
    for arm, label in (("ws", "Warm-start AV"), ("it50", "Suffix-RL iter_50 AV")):
        ps = persample[arm]
        md.append(f"\n**{label}** — cos full {ps['full_cos'][i]:.3f} · "
                  f"first-10 {ps['p10_cos'][i]:.3f} · last-10 {ps['s10_cos'][i]:.3f}\n\n")
        md.append("```\n" + gens[arm][i].strip() + "\n```\n")
open(f"{OUT}/example_outputs.md", "w").write("".join(md))
print("SAMPLES_SWEEP_DONE", flush=True)
