"""Side-by-side AV samples across checkpoints (pre-RL vs quote-penalty iters).

For N held-out av_eval prompts (distinct docs), generate from each checkpoint
with IDENTICAL prompts and per-sample seeds, and write a markdown comparison
with per-sample quote-char counts + a summary table. Sampling is T=1 with
top_p/top_k passed explicitly (never greedy).

Usage:
  python compare_quote_samples.py OUT.md N name1=/path/ckpt1 name2=/path/ckpt2 ...

Checkpoints are HF dirs with nla_meta.yaml. Loads one model at a time.
"""
import re
import sys

import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import INJECT_PLACEHOLDER, extract_explanation_open, normalize_activation

QUOTES = frozenset("\"'`‘’‚‛“”„‟«»‹›「」『』〝〞〟＂＇｀｢｣❛❜❝❞⹂")  # = nla.reward._QUOTE_CHARS
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")
EVAL = "/workspace/out/av_eval.parquet"
SEED0 = 1234

OUT, N = sys.argv[1], int(sys.argv[2])
CKPTS = [a.split("=", 1) for a in sys.argv[3:]]
assert CKPTS, "give at least one name=path checkpoint"

t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
seen = {}
for idx, d in enumerate(docs):
    seen.setdefault(d, idx)
first = list(seen.values())
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
col = t.column
prompts = [col("prompt")[i].as_py() for i in sel]
detok = [col("detokenized_text_truncated")[i].as_py() for i in sel]
cid = [f"{col('doc_id')[i].as_py()}@{col('n_raw_tokens')[i].as_py()}" for i in sel]
vecs = [col("activation_vector")[i].as_py() for i in sel]

def qcount(s: str) -> int:
    return sum(1 for c in s if c in QUOTES)

gens: dict[str, list[str]] = {}
for name, path in CKPTS:
    meta = yaml.safe_load(open(f"{path}/nla_meta.yaml"))
    T = meta["tokens"]
    inj_id, left, right = (T["injection_token_id"], T["injection_left_neighbor_id"],
                           T["injection_right_neighbor_id"])
    inj_char, scale = T["injection_char"], meta["extraction"]["injection_scale"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    emb = model.get_input_embeddings()
    outs = []
    for i in range(N):
        msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)}
                for m in prompts[i]]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        e = emb(ids)
        v = torch.tensor(vecs[i], dtype=torch.float32).view(1, -1)
        e2 = inject_at_marked_positions(ids, e, normalize_activation(v, scale),
                                        inj_id, left, right)
        torch.manual_seed(SEED0 + i)  # same seed per sample across checkpoints
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=e2,
                attention_mask=torch.ones(ids.shape, device=ids.device),
                max_new_tokens=200, do_sample=True, temperature=1.0, top_p=1.0,
                top_k=0, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0], skip_special_tokens=True)
        outs.append(extract_explanation_open(gen) or gen)
        print(f"[{name}] {i + 1}/{N}", flush=True)
    gens[name] = outs
    del model
    torch.cuda.empty_cache()

L = ["# Quote-penalty RL: sample comparison", ""]
L.append(f"{N} held-out `av_eval` prompts (distinct docs), identical prompts + per-sample "
         f"seeds across checkpoints, 200 max new tokens.")
L.append("")
L.append("| checkpoint | quote chars/sample | zero-quote samples | 'final token' echo | CJK |")
L.append("|---|---|---|---|---|")
for name, _ in CKPTS:
    q = [qcount(s) for s in gens[name]]
    echo = sum(1 for s in gens[name] if re.search(r"final token", s, re.I))
    cjk = sum(1 for s in gens[name] if CJK.search(s))
    L.append(f"| {name} | {sum(q)/len(q):.2f} | {sum(1 for c in q if c == 0)}/{N} "
             f"| {echo}/{N} | {cjk}/{N} |")
L.append("")
for i in range(N):
    ctx = detok[i]
    L.append(f"\n---\n\n## Sample {i + 1}/{N} — `{cid[i]}`\n")
    L.append("**Context tail** (activation = last-token L20 state):")
    L.append("```text\n" + ("..." + ctx[-700:] if len(ctx) > 700 else ctx) + "\n```")
    for name, _ in CKPTS:
        s = gens[name][i]
        L.append(f"\n**{name}** (quote chars: {qcount(s)}):")
        L.append("```text\n" + s.strip()[:1500] + "\n```")
open(OUT, "w").write("\n".join(L) + "\n")
print("WROTE", OUT, flush=True)
