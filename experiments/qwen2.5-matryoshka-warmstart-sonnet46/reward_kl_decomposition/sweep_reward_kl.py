"""Per-token-position decomposition of the v3 RL training signal into its two
components: reconstruction reward vs KL penalty.

In the v3 run (run_rl_v3.sh) the two signals enter the loss differently:

  - RECONSTRUCTION is sequence-level: reward = -mse_nrm(critic(prefix), gold)
    on a rollout capped at L ~ U[1,120] content tokens (tokens-mode truncation,
    offset 0 — the trained sequence IS the prefix). Its per-position structure
    is the marginal reward dr(t) = r(t) - r(t-1) of adding token t.
  - KL is per-token: kl_loss_coef * k1_t, k1_t = log pi(x_t) - log pi_ref(x_t)
    at the sampled token (miles --use-kl-loss, default k1 estimator),
    ref = the v3 SFT warm-start, coef 0.03.

Both terms only exist for t <= L, so they share the exposure factor
P(L >= t) = (121-t)/120 and the RAW curves are directly comparable in
reward-units-per-token.

For N held-out distinct-doc prompts this script:
  1. generates one full-length rollout per prompt from the RLed AV at T=1
     (the rollout distribution; capping at L is distributionally identical to
     taking the first L tokens — see nla.truncation "cap, don't post-truncate"),
  2. teacher-forces the same sequence through the policy AND the warm-start
     reference (both with activation injection) -> per-position k1 and
     exact full-vocab KL,
  3. sweeps the critic over every prefix t = 0..len -> per-position reward
     r_i(t), mirroring nla.reward exactly (extract_explanation_open, template,
     -mse_nrm, failed-extraction penalty -2.0).

Outputs in $OUTDIR:
  per_sample.npz   reward [N, Lmax+1] (col t = reward of first-t-token prefix,
                   col 0 = empty prefix), k1 / kl_exact [N, Rmax]
                   (response positions incl. the EOS token when emitted),
                   content_len, resp_len, has_eos; all NaN-padded.
  aggregate.csv    per position t: n_alive, mean_reward, mean_marginal,
                   mean_k1, mean_kl_exact, exposure (P(L>=t) under U[1,120]).
  responses.json   generated texts + per-sample metadata.
"""
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import INJECT_PLACEHOLDER, extract_explanation_open, normalize_activation
from nla_inference import NLACritic

AV_DIR = os.environ.get("AV_DIR", "/workspace/v3/av")
REF_DIR = os.environ.get("REF_DIR", "/workspace/v3ws")
AR_DIR = os.environ.get("AR_DIR", "/workspace/v3/ar")
EVAL = os.environ.get("EVAL", "/workspace/out/v3/av_eval_v3.parquet")
OUTDIR = os.environ.get("OUTDIR", "/workspace/out/rkl")
os.makedirs(OUTDIR, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MAX_NEW = int(os.environ.get("MAX_NEW", "160"))  # v3 ROLLOUT_MAX_RESP
_T = float(os.environ.get("NLA_GEN_TEMP", "1.0"))
GEN_KW = (dict(do_sample=True, temperature=_T, top_p=1.0, top_k=0)
          if _T > 0 else dict(do_sample=False))
_SEED = int(os.environ.get("NLA_GEN_SEED", "0"))
# v3 truncation distribution (run_rl_v3.sh): L ~ U[TRUNC_MIN, TRUNC_MAX] content tokens
TRUNC_MIN = int(os.environ.get("NLA_TRUNC_MIN_TOKENS", "1"))
TRUNC_MAX = int(os.environ.get("NLA_TRUNC_MAX_TOKENS", "120"))
FAILED_REWARD = -2.0  # nla.reward.FAILED_EXTRACTION_REWARD (non-log mode)
dev = "cuda"

meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml"))
T_ = meta["tokens"]
inj_id, left, right = T_["injection_token_id"], T_["injection_left_neighbor_id"], T_["injection_right_neighbor_id"]
inj_char = T_["injection_char"]
inj_scale = meta["extraction"]["injection_scale"]
# The ref forward in training used the same injected embedding contract as the
# policy — a sidecar mismatch would silently compute KL against a different model.
ref_meta = yaml.safe_load(open(f"{REF_DIR}/nla_meta.yaml"))
assert ref_meta["tokens"]["injection_token_id"] == inj_id, "ref/policy marker mismatch"
assert float(ref_meta["extraction"]["injection_scale"]) == float(inj_scale), \
    f"ref injection_scale {ref_meta['extraction']['injection_scale']} != policy {inj_scale}"
assert "<explanation>" not in open(f"{AV_DIR}/nla_meta.yaml").read(), \
    "this script assumes the v3 UNTAGGED format (content tokens == response tokens)"

tok = AutoTokenizer.from_pretrained(AV_DIR)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
# Positions align only if critic and AV tokenize identically (both Qwen2.5
# here). Checked up front so a mismatch fails in seconds, not after the sweep.
assert AutoTokenizer.from_pretrained(AR_DIR).get_vocab() == tok.get_vocab(), \
    "AV/AR tokenizer mismatch — prefix positions would not align"

# ---- select N distinct-doc samples (same logic as ../fve_truncation_sweep) ----
t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
seen = {}
for idx, d in enumerate(docs):
    if d not in seen:
        seen[d] = idx
first = list(seen.values())
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
col = t.column
prompts = [col("prompt")[i].as_py() for i in sel]
vecs = [np.asarray(col("activation_vector")[i].as_py(), dtype=np.float32) for i in sel]
print(f"selected {len(sel)} samples ({len(first)} distinct docs in {EVAL})", flush=True)

seqs = []  # prompt token ids per sample (chat template, generation prompt)
for p in prompts:
    msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in p]
    seqs.append(tok.apply_chat_template(msgs, add_generation_prompt=True))
V_norm = torch.stack([
    normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), inj_scale)[0]
    for v in vecs
])

# ─────────────────────── phase 1: policy rollouts at T=1 ───────────────────────
av = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16).to(dev).eval()
emb = av.get_input_embeddings()


@torch.no_grad()
def av_generate(idx):
    batch = [seqs[i] for i in idx]
    m = max(len(s) for s in batch)
    pad = tok.pad_token_id
    inp = np.full((len(batch), m), pad, dtype=np.int64)
    att = np.zeros((len(batch), m), dtype=np.int64)
    for k, s in enumerate(batch):
        inp[k, m - len(s):] = s
        att[k, m - len(s):] = 1
    inp = torch.tensor(inp, device=dev)
    att = torch.tensor(att, device=dev)
    e = emb(inp)
    out = av.generate(
        inputs_embeds=inject_at_marked_positions(inp, e, V_norm[idx].to(e.dtype), inj_id, left, right),
        attention_mask=att, max_new_tokens=MAX_NEW, pad_token_id=pad, **GEN_KW,
    )
    return out.tolist()  # generated tokens only (inputs_embeds path)


CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")
# generate() stops on any of these; batched rows are pad-filled past the first one
_eos_cfg = av.generation_config.eos_token_id
EOS_IDS = set(_eos_cfg if isinstance(_eos_cfg, list) else [_eos_cfg]) | {tok.eos_token_id}
torch.manual_seed(_SEED)
resp_ids = []   # per sample: generated ids up to AND INCLUDING eos (if emitted)
has_eos = []
ncjk = 0
BG = 16
for s in range(0, len(sel), BG):
    idx = torch.arange(s, min(s + BG, len(sel)))
    for row in av_generate(idx):
        eos_at = next((k for k, x in enumerate(row) if x in EOS_IDS), None)
        if eos_at is not None:
            row = row[: eos_at + 1]
            has_eos.append(True)
        else:
            has_eos.append(False)
        resp_ids.append(row)
        if CJK.search(tok.decode(row, skip_special_tokens=True)):
            ncjk += 1
    print(f"  AV gen {min(s + BG, len(sel))}/{len(sel)} (cjk={ncjk})", flush=True)
content_ids = [r[:-1] if e else r for r, e in zip(resp_ids, has_eos)]
content_len = [len(c) for c in content_ids]
resp_len = [len(r) for r in resp_ids]
texts = [tok.decode(c, skip_special_tokens=True) for c in content_ids]
print(f"gen done: content len median={int(np.median(content_len))} "
      f"max={max(content_len)} eos={sum(has_eos)}/{N} cjk={ncjk}", flush=True)

# ─────────────────── phase 2: per-position KL vs warm-start ref ─────────────────
ref = AutoModelForCausalLM.from_pretrained(REF_DIR, dtype=torch.bfloat16).to(dev).eval()
ref_emb = ref.get_input_embeddings()
Rmax = max(resp_len)
k1_mat = np.full((N, Rmax), np.nan, dtype=np.float32)
klx_mat = np.full((N, Rmax), np.nan, dtype=np.float32)


@torch.no_grad()
def resp_logprobs(model, embedder, ids, i, P, R):
    """Log-softmax over the R response positions of sample i. [R, V] fp32."""
    e = embedder(ids)
    e = inject_at_marked_positions(ids, e, V_norm[i : i + 1].to(e.dtype), inj_id, left, right)
    logits = model(inputs_embeds=e, use_cache=False).logits[0, P - 1 : P - 1 + R]
    return torch.log_softmax(logits.float(), dim=-1)


for i in range(N):
    full = seqs[i] + resp_ids[i]
    P, R = len(seqs[i]), resp_len[i]
    ids = torch.tensor([full], device=dev)
    lp_pol = resp_logprobs(av, emb, ids, i, P, R)
    lp_ref = resp_logprobs(ref, ref_emb, ids, i, P, R)
    xt = torch.tensor(resp_ids[i], device=dev)
    k1_mat[i, :R] = (lp_pol.gather(1, xt[:, None]) - lp_ref.gather(1, xt[:, None]))[:, 0].cpu().numpy()
    klx_mat[i, :R] = (lp_pol.exp() * (lp_pol - lp_ref)).sum(-1).cpu().numpy()
    if (i + 1) % 25 == 0:
        print(f"  KL {i + 1}/{N}", flush=True)
print(f"KL done: mean k1={np.nanmean(k1_mat):.4f}  mean exact={np.nanmean(klx_mat):.4f}", flush=True)

del av, emb, ref, ref_emb
torch.cuda.empty_cache()

# ──────────────── phase 3: critic reward per prefix length ────────────────
critic = NLACritic(AR_DIR, device=dev)
from safetensors.torch import load_file  # noqa: E402
assert torch.isfinite(load_file(f"{AR_DIR}/value_head.safetensors")["weight"]).all(), \
    "corrupt value head (see MODEL_USAGE.md)"
ms = critic.mse_scale
ctok = critic.tokenizer
ctok.padding_side = "right"
if ctok.pad_token_id is None:
    ctok.pad_token = ctok.eos_token
G = torch.stack([torch.tensor(v) for v in vecs]).float()
gn = normalize_activation(G, ms)

# Mirror nla.reward exactly: extract_explanation_open on the decoded prefix
# (None -> failed-extraction penalty), critic template, -mse on normalized vecs.
jobs = []  # (sample_i, t, explanation | None)
for i in range(N):
    for tt in range(0, content_len[i] + 1):
        jobs.append((i, tt, extract_explanation_open(tok.decode(content_ids[i][:tt], skip_special_tokens=True))))
Lmax = max(content_len)
reward_mat = np.full((N, Lmax + 1), np.nan, dtype=np.float32)

for i, tt, e in jobs:
    if e is None:  # contentless prefix -> failed-extraction penalty (nla.reward)
        reward_mat[i, tt] = FAILED_REWARD
live = [(i, tt, e) for i, tt, e in jobs if e is not None]
BS = 64
for b in range(0, len(live), BS):
    chunk = live[b : b + BS]
    enc = ctok([critic.template.format(explanation=e) for _, _, e in chunk],
               return_tensors="pt", add_special_tokens=True, padding=True)
    ids = enc["input_ids"].to(dev)
    am = enc["attention_mask"].to(dev)
    with torch.no_grad():
        hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
        h = hs[torch.arange(hs.size(0)), am.sum(1) - 1]
        pred = critic.value_head(h).float().cpu()
    pn = normalize_activation(pred, ms)
    for (i, tt, _), p in zip(chunk, pn):
        mse = ((p - gn[i]) ** 2).mean().item()
        reward_mat[i, tt] = -mse if np.isfinite(mse) else FAILED_REWARD
    if (b // BS) % 20 == 0:
        print(f"  critic {b}/{len(live)}", flush=True)
print(f"critic done: mean full-length reward="
      f"{np.mean([reward_mat[i, content_len[i]] for i in range(N)]):.4f}", flush=True)

# ─────────────────────────── aggregate + save ───────────────────────────
np.savez_compressed(
    f"{OUTDIR}/per_sample.npz", reward=reward_mat, k1=k1_mat, kl_exact=klx_mat,
    content_len=np.array(content_len), resp_len=np.array(resp_len),
    has_eos=np.array(has_eos), gen_temp=_T, seed=_SEED,
    trunc_min=TRUNC_MIN, trunc_max=TRUNC_MAX,
)
json.dump([{"i": i, "doc_row": sel[i], "text": texts[i], "content_len": content_len[i],
            "has_eos": bool(has_eos[i])} for i in range(N)],
          open(f"{OUTDIR}/responses.json", "w"), indent=1)

with open(f"{OUTDIR}/aggregate.csv", "w") as fh:
    fh.write("t,n_alive,mean_reward,mean_marginal,mean_k1,mean_kl_exact,exposure\n")
    for tt in range(1, Lmax + 1):
        alive = [i for i in range(N) if content_len[i] >= tt]
        r = np.mean([reward_mat[i, tt] for i in alive])
        dr = np.mean([reward_mat[i, tt] - reward_mat[i, tt - 1] for i in alive])
        k1 = np.nanmean(k1_mat[:, tt - 1])
        kx = np.nanmean(klx_mat[:, tt - 1])
        expo = max(0.0, min(1.0, (TRUNC_MAX - tt + 1) / (TRUNC_MAX - TRUNC_MIN + 1)))
        fh.write(f"{tt},{len(alive)},{r:.6f},{dr:.6f},{k1:.6f},{kx:.6f},{expo:.6f}\n")
# EOS position stats (KL exists there but no reconstruction marginal)
eos_k1 = [k1_mat[i, resp_len[i] - 1] for i in range(N) if has_eos[i]]
print(f"EOS-position k1: mean={np.mean(eos_k1):.4f} (n={len(eos_k1)})" if eos_k1 else "no EOS emitted")
print("RKL_DONE", flush=True)
