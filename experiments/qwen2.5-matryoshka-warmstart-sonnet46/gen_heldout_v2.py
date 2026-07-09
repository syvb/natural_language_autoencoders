"""Verbalize N distinct held-out activations with the v2 RL AV; write a markdown dump.

Loads the v2 AV (converted HF, from_pretrained-ready), reads av_eval.parquet (raw activations,
norm=none), picks N distinct doc_ids, injects each at the marker and greedily decodes, then writes
markdown with the input-context tail + the verbalized list items (per-item token length) + CJK flag.

Env: AV_DIR (HF AV dir w/ nla_meta.yaml), EVAL (av_eval.parquet), OUT (md path). Arg: N (default 100).
"""
import os, re, sys, yaml, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open

AV = os.environ["AV_DIR"]; EVAL = os.environ["EVAL"]; OUT = os.environ.get("OUT", "/workspace/heldout_v2_100.md")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MAXNEW = 384; BATCH = 10
meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = meta["extraction"]["injection_scale"]
_pt = meta["prompt_templates"]; actor = _pt.get("actor") or _pt["av"]
tok = AutoTokenizer.from_pretrained(AV)
tok.padding_side = "left"
model = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = model.get_input_embeddings()
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿＀-￯]")

t = pq.read_table(EVAL); cols = t.column_names
acts = t.column("activation_vector").to_pylist()
docs = t.column("doc_id").to_pylist() if "doc_id" in cols else list(range(len(acts)))
ctx = t.column("detokenized_text_truncated").to_pylist() if "detokenized_text_truncated" in cols else [None] * len(acts)
seen = {}
for idx, d in enumerate(docs):
    seen.setdefault(d, idx)
first = list(seen.values()); step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]

ptext = tok.apply_chat_template([{"role": "user", "content": actor.format(injection_char=inj_char)}],
                                add_generation_prompt=True, tokenize=False)
pids = tok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
S = pids.shape[1]


@torch.no_grad()
def gen_batch(idxs):
    n = len(idxs)
    ids = pids.repeat(n, 1)
    v = torch.tensor([acts[i] for i in idxs], dtype=torch.float32, device="cuda")
    e = inject_at_marked_positions(ids, emb(ids), normalize_activation(v, scale), inj_id, left, right)
    out = model.generate(inputs_embeds=e, attention_mask=torch.ones(n, S, device="cuda"),
                         max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.eos_token_id)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


L = [f"# v2 RL NLA (iter_200) — {N} held-out AV verbalizations\n",
     "Model: `syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v2-rl` (iter_0000200). Greedy, 384-token "
     "cap. Each held-out activation (raw, from `av_eval.parquet`) is injected at the marker and the AV's "
     "explanation list is decoded. `detokenized_text_truncated` = the source text whose layer-20 last-token "
     "activation was captured.\n"]
ncjk = 0
for s in range(0, len(sel), BATCH):
    chunk = sel[s:s + BATCH]
    for idx, g in zip(chunk, gen_batch(chunk)):
        expl = extract_explanation_open(g) or g
        items = [ln.strip() for ln in re.split(r"\n+", expl.strip()) if ln.strip()]
        cjk = bool(CJK.search(g)); ncjk += cjk
        lens = [len(tok(it, add_special_tokens=False)["input_ids"]) for it in items]
        n = s + chunk.index(idx)
        L.append(f"## {n}. doc `{docs[idx]}`  ({len(items)} items, CJK={cjk})\n")
        if ctx[idx]:
            L.append(f"*source text (tail):* …{str(ctx[idx])[-320:]}\n")
        body = "\n".join(f"[{ln:2d} tok] {it}" for ln, it in zip(lens, items))
        L.append(f"```\n{body}\n```\n")
    print(f"  {min(s+BATCH,len(sel))}/{len(sel)}", flush=True)
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT} | {N} samples | CJK {ncjk}/{N}", flush=True)
print("GEN_HELDOUT_DONE", flush=True)
