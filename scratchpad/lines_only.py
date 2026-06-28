"""Lines-only re-run: AV-generate then line-truncation FVE sweep (coalesced newlines)."""
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


expls = []
for s in range(0, len(sel), 16):
    for txt in av_generate(prompts[s:s+16], vecs[s:s+16]):
        expls.append(extract_explanation_open(txt) or txt)
    print(f"  AV gen {min(s+16,len(sel))}/{len(sel)}", flush=True)

del av, emb; torch.cuda.empty_cache()
critic = NLACritic(AR_DIR, device=dev); ms = critic.mse_scale
G = torch.stack([torch.tensor(v) for v in vecs]).float()
ctok = critic.tokenizer; ctok.padding_side = "right"
if ctok.pad_token_id is None: ctok.pad_token = ctok.eos_token


@torch.no_grad()
def reconstruct_batch(explanations, bs=32):
    outs = []
    for i in range(0, len(explanations), bs):
        prompts = [critic.template.format(explanation=e) for e in explanations[i:i+bs]]
        enc = ctok(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
        hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
        h = hs[torch.arange(hs.size(0)), am.sum(1) - 1]
        outs.append(critic.value_head(h).float().cpu())
    return torch.cat(outs, 0)


def fve(P, G):
    gn = G / G.norm(dim=-1, keepdim=True) * ms
    pn = P / P.norm(dim=-1, keepdim=True) * ms
    mu = gn.mean(0)
    return 1 - ((pn - gn) ** 2).mean().item() / ((gn - mu) ** 2).mean().item()


# coalesce repeated newlines; drop blank lines
lines_per = [[p for p in re.split(r"\n+", e.strip()) if p.strip()] for e in expls]
Kmax = max(len(x) for x in lines_per); klens = [len(x) for x in lines_per]
print(f"LINES sweep (coalesced): Kmax={Kmax}  median_lines={int(np.median(klens))}", flush=True)
rows = []
for K in range(1, Kmax + 1):
    P = reconstruct_batch(["\n".join(lp[:K]) for lp in lines_per])
    f = fve(P, G); nfull = sum(1 for x in klens if x <= K)
    rows.append((K, f, nfull)); print(f"  K={K:>2}  FVE={f:.4f}  ({nfull}/{N} full)", flush=True)
with open(f"{OUTDIR}/lines_fve.csv", "w") as fh:
    fh.write("length_lines,fve,n_samples_at_full\n")
    for K, f, n in rows: fh.write(f"{K},{f:.6f},{n}\n")
print("LINES_DONE", flush=True)
