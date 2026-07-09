"""Per-truncation-length round-trip FVE sweep (tokens AND lines) for the AV/AR pair.

For 100 held-out distinct-doc samples:
  AV (transformers+injection): gold activation v -> explanation text (greedy, full)
For each truncation length L (1..max):
  truncate AV explanation to first L content tokens (or first L lines), feed to
  AR (NLACritic.reconstruct) -> v_hat; FVE over the 100 = 1 - dirMSE/var
  (both pred+gold L2-normalized to mse_scale, raw-mean baseline). Identical
  FVE def to eval_round_trip_fve.py's stats().

Writes token_fve.csv, lines_fve.csv and PNG plots.
"""
import os, sys, re, yaml, torch, numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, INJECT_PLACEHOLDER, extract_explanation_open
from nla_inference import NLACritic

AV_DIR = os.environ.get("AV_DIR", "/workspace/hf_out/av")
AR_DIR = os.environ.get("AR_DIR", "/workspace/hf_out/ar")
EVAL = os.environ.get("EVAL", "/workspace/out/av_eval.parquet")
OUTDIR = os.environ.get("OUTDIR", "/workspace/sweep"); os.makedirs(OUTDIR, exist_ok=True)
TAG = os.environ.get("TAG", "kl0.01/iter_0000200")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
dev = "cuda"

meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml"))
T = meta["tokens"]; inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]
right = T["injection_right_neighbor_id"]; inj_char = T["injection_char"]
inj_scale = meta["extraction"]["injection_scale"]

tok = AutoTokenizer.from_pretrained(AV_DIR); tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
av = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16).to(dev).eval()
emb = av.get_input_embeddings()

# --- select same 100 distinct-doc samples as gen_md_samples.py (first occ, stride) ---
t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
seen = {}
for idx, d in enumerate(docs):
    if d not in seen: seen[d] = idx
first = list(seen.values())
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
col = t.column
prompts = [col("prompt")[i].as_py() for i in sel]
vecs = [np.asarray(col("activation_vector")[i].as_py(), dtype=np.float32) for i in sel]
print(f"selected {len(sel)} samples", flush=True)


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
                      attention_mask=att, max_new_tokens=256, do_sample=False, pad_token_id=pad)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")
expls = []; ncjk = 0
BG = 16
for s in range(0, len(sel), BG):
    for txt in av_generate(prompts[s:s+BG], vecs[s:s+BG]):
        if CJK.search(txt): ncjk += 1
        expls.append(extract_explanation_open(txt) or txt)
    print(f"  AV gen {min(s+BG,len(sel))}/{len(sel)} (cjk={ncjk})", flush=True)

# free AV to give the critic memory headroom for big reconstruct batches
del av, emb; torch.cuda.empty_cache()
critic = NLACritic(AR_DIR, device=dev)
ms = critic.mse_scale
G = torch.stack([torch.tensor(v) for v in vecs]).float()
ctok = critic.tokenizer
ctok.padding_side = "right"
if ctok.pad_token_id is None: ctok.pad_token = ctok.eos_token


@torch.no_grad()
def reconstruct_batch(explanations, bs=32):
    outs = []
    for i in range(0, len(explanations), bs):
        chunk = explanations[i:i+bs]
        prompts = [critic.template.format(explanation=e) for e in chunk]
        enc = ctok(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
        hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
        last = am.sum(1) - 1
        h = hs[torch.arange(hs.size(0)), last]
        outs.append(critic.value_head(h).float().cpu())
    return torch.cat(outs, 0)


def fve(P, G):
    gn = G / G.norm(dim=-1, keepdim=True) * ms
    pn = P / P.norm(dim=-1, keepdim=True) * ms
    mu = gn.mean(0)
    return 1 - ((pn - gn) ** 2).mean().item() / ((gn - mu) ** 2).mean().item()


# ---- token-truncation IDs per sample ----
tok_ids = [ctok.encode(e, add_special_tokens=False) for e in expls]  # critic-tokenizer content tokens
Lmax = max(len(x) for x in tok_ids)
lens = [len(x) for x in tok_ids]
print(f"\nTOKEN sweep: Lmax={Lmax}  median_len={int(np.median(lens))}  min={min(lens)}", flush=True)
tok_rows = []
for L in range(1, Lmax + 1):
    trunc = [ctok.decode(ids[:L]) for ids in tok_ids]   # full text if shorter than L
    P = reconstruct_batch(trunc)
    f = fve(P, G)
    nfull = sum(1 for x in lens if x <= L)
    tok_rows.append((L, f, nfull))
    if L % 10 == 0 or L == 1 or L == Lmax:
        print(f"  L={L:>3}  FVE={f:.4f}  ({nfull}/{N} at full len)", flush=True)

# ---- line-truncation per sample (coalesce repeated newlines; drop blank lines) ----
lines_per = [[p for p in re.split(r"\n+", e.strip()) if p.strip()] for e in expls]
Kmax = max(len(x) for x in lines_per)
klens = [len(x) for x in lines_per]
print(f"\nLINES sweep: Kmax={Kmax}  median_lines={int(np.median(klens))}", flush=True)
line_rows = []
for K in range(1, Kmax + 1):
    trunc = ["\n".join(lp[:K]) for lp in lines_per]
    P = reconstruct_batch(trunc)
    f = fve(P, G)
    nfull = sum(1 for x in klens if x <= K)
    line_rows.append((K, f, nfull))
    print(f"  K={K:>2}  FVE={f:.4f}  ({nfull}/{N} at full lines)", flush=True)

# full-length reference
fve_full = fve(reconstruct_batch([e for e in expls]), G)

with open(f"{OUTDIR}/token_fve.csv", "w") as fh:
    fh.write("length_tokens,fve,n_samples_at_full\n")
    for L, f, n in tok_rows: fh.write(f"{L},{f:.6f},{n}\n")
with open(f"{OUTDIR}/lines_fve.csv", "w") as fh:
    fh.write("length_lines,fve,n_samples_at_full\n")
    for K, f, n in line_rows: fh.write(f"{K},{f:.6f},{n}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
xs = [r[0] for r in tok_rows]; ys = [r[1] for r in tok_rows]
a1.plot(xs, ys, lw=1.6, color="#1f77b4")
a1.axhline(fve_full, ls="--", color="gray", lw=1, label=f"full-len FVE={fve_full:.3f}")
a1.set_xlabel("AV explanation truncation length (content tokens)"); a1.set_ylabel("round-trip FVE")
a1.set_title(f"FVE vs token-truncation length (N={N})"); a1.grid(alpha=.3); a1.legend()
xk = [r[0] for r in line_rows]; yk = [r[1] for r in line_rows]
a2.plot(xk, yk, marker="o", ms=4, lw=1.6, color="#d62728")
a2.axhline(fve_full, ls="--", color="gray", lw=1, label=f"full-len FVE={fve_full:.3f}")
a2.set_xlabel("AV explanation truncation length (lines)"); a2.set_ylabel("round-trip FVE")
a2.set_title(f"FVE vs line-truncation length (N={N})"); a2.grid(alpha=.3); a2.legend()
fig.suptitle(f"{TAG} — round-trip FVE vs AV-explanation truncation", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fve_vs_truncation.png", dpi=130, bbox_inches="tight")
print(f"\nfull-len FVE={fve_full:.4f}  cjk={ncjk}/{N}")
print("SWEEP_DONE", flush=True)
