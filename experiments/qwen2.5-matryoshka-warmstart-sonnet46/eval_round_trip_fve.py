"""End-to-end round-trip FVE for the AV/AR pair, with SHORT-PREFIX breakdown.

For each held-out sample:  original activation v
  AV (transformers + injection):  v -> explanation text (greedy, full length)
  AR (NLACritic.reconstruct):     text -> predicted activation v_hat
FVE over the set:  1 - MSE(norm(v_hat), norm(v)) / MSE(norm(v), mu)  (raw-mean baseline)

The truncated-RL success metric is INFORMATION UPFRONT: how well the critic
reconstructs from only the FIRST L content tokens of the AV explanation. We
report round-trip FVE at several fixed L (default 10/30/60/130) plus full
length. Over RL training, short-L FVE should RISE (and the gap to full-length
FVE should shrink) as the model front-loads the important content.

Also reports the critic-only reference: AR fed the GOLD Sonnet text (full).

Usage:  python eval_round_trip_fve.py [N] [L1,L2,...]
  N   number of held-out samples (default 300)
  Ls  comma-separated prefix lengths in CONTENT tokens (default 10,30,60,130)
Env:  AV_DIR, AR_DIR, EVAL override the default /workspace paths.
"""
import os, sys, yaml, torch, numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, INJECT_PLACEHOLDER, extract_explanation_open

# nla_inference lives at the repo root; import works when run from there (the
# README's eval invocation cds to /root and sets PYTHONPATH=/workspace/nla).
from nla_inference import NLACritic

AV_DIR = os.environ.get("AV_DIR", "/workspace/av")
AR_DIR = os.environ.get("AR_DIR", "/workspace/ar")
EVAL = os.environ.get("EVAL", "/workspace/av_eval.parquet")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PREFIX_LENS = (
    [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
    else [10, 30, 60, 130]
)
BATCH = 16
dev = "cuda"

meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml"))
T = meta["tokens"]; inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]
right = T["injection_right_neighbor_id"]; inj_char = T["injection_char"]
inj_scale = meta["extraction"]["injection_scale"]
print(f"AV: inj_char={inj_char!r} id={inj_id} neighbors=({left},{right}) injection_scale={inj_scale}", flush=True)
print(f"prefix lengths (content tokens): {PREFIX_LENS} + full", flush=True)

tok = AutoTokenizer.from_pretrained(AV_DIR); tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
av = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16).to(dev).eval()
emb = av.get_input_embeddings()
critic = NLACritic(AR_DIR, device=dev)
ms = critic.mse_scale
print(f"mse_scale={ms}", flush=True)

t = pq.read_table(EVAL)
# distinct docs for diversity
docs = t.column("doc_id").to_pylist()
seen = {}
for i, d in enumerate(docs):
    seen.setdefault(d, []).append(i)
order = []
# round-robin one position per doc until we have N
docl = list(seen.values()); k = 0
while len(order) < N and any(k < len(v) for v in docl):
    for v in docl:
        if k < len(v): order.append(v[k])
        if len(order) >= N: break
    k += 1
order = order[:N]
col = t.column
prompts = [col("prompt")[i].as_py() for i in order]
resp = [col("response")[i].as_py() for i in order]
vecs = [np.asarray(col("activation_vector")[i].as_py(), dtype=np.float32) for i in order]
print(f"evaluating {len(order)} samples from {len(set(docs[i] for i in order))} docs", flush=True)


def first_l_content_tokens(text, L):
    """First L CONTENT tokens of `text`, re-decoded — mirrors the training-time
    content-token cap so eval truncation == train truncation."""
    ids = tok.encode(text, add_special_tokens=False)[:L]
    return tok.decode(ids)


@torch.no_grad()
def av_generate(batch_prompts, batch_vecs):
    seqs = []
    for p in batch_prompts:
        msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in p]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        seqs.append(ids)
    m = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), m), pad, dtype=np.int64); att = np.zeros((len(seqs), m), dtype=np.int64)
    for k, s in enumerate(seqs):
        inp[k, m - len(s):] = s; att[k, m - len(s):] = 1   # LEFT pad
    inp = torch.tensor(inp, device=dev); att = torch.tensor(att, device=dev)
    e = emb(inp)
    V = torch.stack([normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), inj_scale)[0] for v in batch_vecs])
    e2 = inject_at_marked_positions(inp, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=att, max_new_tokens=256,
                      do_sample=False, pad_token_id=pad)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


import re
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")
golds = []
preds_gold = []                              # critic-only, full gold text
preds_rt = {L: [] for L in PREFIX_LENS}      # round-trip at each prefix length
preds_rt["full"] = []                        # round-trip, full AV text
ncjk = 0
for s in range(0, len(order), BATCH):
    bp = prompts[s:s + BATCH]; bv = vecs[s:s + BATCH]
    texts = av_generate(bp, bv)
    for j, txt in enumerate(texts):
        idx = s + j
        if CJK.search(txt): ncjk += 1
        av_expl = extract_explanation_open(txt) or txt
        gold_expl = extract_explanation_open(resp[idx]) or resp[idx]
        preds_rt["full"].append(critic.reconstruct(av_expl))
        for L in PREFIX_LENS:
            preds_rt[L].append(critic.reconstruct(first_l_content_tokens(av_expl, L)))
        preds_gold.append(critic.reconstruct(gold_expl))
        golds.append(torch.tensor(vecs[idx]))
    print(f"  {min(s + BATCH, len(order))}/{len(order)}  (cjk so far={ncjk})", flush=True)

G = torch.stack(golds).float()


def stats(P, G, ms):
    gn = G / G.norm(dim=-1, keepdim=True) * ms
    pn = P / P.norm(dim=-1, keepdim=True) * ms
    mu = gn.mean(0)
    num = ((pn - gn) ** 2).mean().item()
    den = ((gn - mu) ** 2).mean().item()
    cos = ((pn * gn).sum(-1) / (pn.norm(dim=-1) * gn.norm(dim=-1))).mean().item()
    return 1 - num / den, num, cos, den


fve_g, mse_g, cos_g, den = stats(torch.stack(preds_gold).float(), G, ms)
print("\n" + "=" * 72)
print(f"N={len(order)}  mse_scale={ms:.3f}  raw-mean baseline (denom)={den:.4f}")
print(f"CJK-in-AV-output: {ncjk}/{len(order)}")
print("-" * 72)
print("INFORMATION-UPFRONT — round-trip FVE by AV content-token prefix length:")
print(f"  {'prefix':>8}  {'FVE':>8}  {'dir-MSE':>8}  {'cos':>6}")
for L in PREFIX_LENS:
    fve, mse, cos, _ = stats(torch.stack(preds_rt[L]).float(), G, ms)
    print(f"  {L:>8}  {fve:>8.4f}  {mse:>8.4f}  {cos:>6.3f}")
fve_full, mse_full, cos_full, _ = stats(torch.stack(preds_rt["full"]).float(), G, ms)
print(f"  {'full':>8}  {fve_full:>8.4f}  {mse_full:>8.4f}  {cos_full:>6.3f}")
print("-" * 72)
print(f"CRITIC-ONLY (gold Sonnet text, full -> AR):  FVE={fve_g:.4f}  dir-MSE={mse_g:.4f}  cos={cos_g:.4f}")
print("=" * 72)
print("FVE_EVAL_DONE", flush=True)
