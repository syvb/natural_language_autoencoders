"""Per-example FVE-by-line for the v2 NLA round-trip.

For each of a few random held-out activations v:
  AV (v2):  v -> explanation list (greedy)           [lines = newline items]
  AR (v2):  for k=1..N, reconstruct from the FIRST k LINES -> v_hat_k
  FVE_k = 1 - ||norm(v_hat_k) - norm(v)||^2 / ||norm(v) - mu||^2
          (per-example; mu = population mean of normalized held-out golds)

Both v and v_hat are L2-normalized to mse_scale before the metric (the training
convention). Writes per-example JSON (lines + fve/cos per prefix) for local plotting.

Env: AV_DIR, AR_DIR, EVAL. Args: N_EXAMPLES (default 6) SEED (default 0).
"""
import json, os, re, sys, yaml, torch
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open
from nla_inference import NLACritic

AV_DIR = os.environ["AV_DIR"]; AR_DIR = os.environ["AR_DIR"]; EVAL = os.environ["EVAL"]
OUT = os.environ.get("OUT", "/workspace/fve_by_line.json")
N_EX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
import os as _os
_T = float(_os.environ.get("NLA_GEN_TEMP", "0"))
GEN_KW = (dict(do_sample=True, temperature=_T, top_p=1.0, top_k=0)
          if _T > 0 else dict(do_sample=False))
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAXNEW = 384
dev = "cuda"

meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml")); T = meta["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = meta["extraction"]["injection_scale"]
actor = (meta["prompt_templates"].get("actor") or meta["prompt_templates"]["av"])
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿＀-￯]")

# ---- data + population mean (mu) of normalized golds ----
t = pq.read_table(EVAL)
acts = t.column("activation_vector")
docs = t.column("doc_id").to_pylist()
ctx = t.column("detokenized_text_truncated").to_pylist() if "detokenized_text_truncated" in t.column_names else [None]*len(docs)
allv = np.asarray(acts.to_pylist(), dtype=np.float32)
G = torch.from_numpy(allv)
ms = None  # set after critic loads

# pick N_EX random distinct docs
seen = {}
for i, d in enumerate(docs):
    seen.setdefault(d, i)
first = list(seen.values())
rng = np.random.RandomState(SEED)
sel = sorted(rng.choice(first, size=min(N_EX, len(first)), replace=False).tolist())
print(f"selected docs: {[docs[i] for i in sel]}", flush=True)

# ---- AV: generate explanations for the selected examples ----
tok = AutoTokenizer.from_pretrained(AV_DIR)
av = Qwen2ForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
ptext = tok.apply_chat_template([{"role": "user", "content": actor.format(injection_char=inj_char)}],
                                add_generation_prompt=True, tokenize=False)
pids = tok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)


# Batched generation is safe here: every row uses the SAME prompt (identical
# length — no padding asymmetry). GEN_BS=1 reproduces the original loop.
GEN_BS = int(_os.environ.get("NLA_GEN_BS", "1"))


@torch.no_grad()
def av_gen(idxs):
    n = len(idxs)
    ids = pids.repeat(n, 1)
    v = torch.tensor(np.stack([allv[i] for i in idxs]), dtype=torch.float32, device=dev)
    e = inject_at_marked_positions(ids, emb(ids), normalize_activation(v, scale), inj_id, left, right)
    out = av.generate(inputs_embeds=e, attention_mask=torch.ones_like(ids),
                      max_new_tokens=MAXNEW, **GEN_KW, pad_token_id=tok.eos_token_id)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


expls = {}
for s in range(0, len(sel), GEN_BS):
    chunk = sel[s:s + GEN_BS]
    for idx, g in zip(chunk, av_gen(chunk)):
        expl = extract_explanation_open(g) or g
        lines = [ln.strip() for ln in re.split(r"\n+", expl.strip()) if ln.strip()]
        expls[idx] = (lines, bool(CJK.search(g)))
        print(f"  doc {docs[idx]}: {len(lines)} lines, cjk={expls[idx][1]}", flush=True)
del av, emb
torch.cuda.empty_cache()

# ---- AR critic: reconstruct from each line-prefix ----
critic = NLACritic(AR_DIR, device=dev)
ms = critic.mse_scale
# population mean of normalized golds
Gn = G / G.norm(dim=-1, keepdim=True).clamp_min(1e-12) * ms
mu = Gn.mean(0)
print(f"mse_scale={ms:.3f}  population mu computed over {len(G)} golds", flush=True)


def fve_one(gold_raw, pred_raw):
    gn = gold_raw / gold_raw.norm().clamp_min(1e-12) * ms
    pn = pred_raw / pred_raw.norm().clamp_min(1e-12) * ms
    den = ((gn - mu) ** 2).sum().item()
    num = ((pn - gn) ** 2).sum().item()
    cos = (pn @ gn / (pn.norm() * gn.norm())).item()
    return 1 - num / den, cos


out_rows = []
for idx in sel:
    lines, cjk = expls[idx]
    gold = torch.from_numpy(allv[idx]).float()
    fves, coss = [], []
    for k in range(1, len(lines) + 1):
        prefix = "\n".join(lines[:k])
        pred = critic.reconstruct(prefix)
        f, c = fve_one(gold, pred)
        fves.append(round(f, 4)); coss.append(round(c, 4))
    out_rows.append({"doc_id": docs[idx], "n_lines": len(lines), "cjk": cjk,
                     "source_tail": (str(ctx[idx])[-400:] if ctx[idx] else None),
                     "lines": lines, "fve_by_k": fves, "cos_by_k": coss})
    print(f"  doc {docs[idx]}: FVE 1-line={fves[0]:.3f}  full={fves[-1]:.3f}  (n={len(lines)})", flush=True)

json.dump({"mse_scale": ms, "seed": SEED, "examples": out_rows}, open(OUT, "w"), indent=1)
print(f"wrote {OUT}", flush=True)
print("FVE_BY_LINE_DONE", flush=True)
