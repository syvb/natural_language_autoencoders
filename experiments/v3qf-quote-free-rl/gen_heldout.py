"""Verbalize N distinct held-out activations with the v3qf AV; write a markdown dump.

Loads the v3qf AV (converted HF, from_pretrained-ready), reads av_eval_v3.parquet
(raw activations, norm=none, v3 format), picks N distinct doc_ids, injects each at
the marker and decodes at temperature 1 (top_p=1, top_k=0 — the AV generation
convention), then writes markdown with the input-context tail + the verbalized
bullet lines (per-line token length) + the penalized-quote-char count + CJK flag.

Env: AV_DIR (HF AV dir w/ nla_meta.yaml), EVAL (av_eval_v3.parquet), OUT (md path).
Arg: N (default 50).
"""
import ast, os, re, sys, pathlib, yaml, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open


def _load_quote_chars() -> frozenset:
    """The penalized set from nla/reward.py WITHOUT importing it (the module
    pulls ray/miles, absent on lightweight inference boxes). AST-read so this
    can never drift from what the reward counted."""
    src = pathlib.Path(__file__).resolve().parents[2] / "nla" / "reward.py"
    for node in ast.walk(ast.parse(src.read_text())):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "_QUOTE_CHARS" for t in node.targets)):
            return frozenset(ast.literal_eval(node.value.args[0]))
    raise RuntimeError(f"_QUOTE_CHARS not found in {src}")


_QUOTE_CHARS = _load_quote_chars()

AV = os.environ["AV_DIR"]; EVAL = os.environ["EVAL"]; OUT = os.environ.get("OUT", "/workspace/heldout_v3qf.md")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
# The v3 recipe never trains EOS, so generation always runs to the cap; 120 is
# the training-time content-budget max (~U[1,120]).
MAXNEW = 120; BATCH = 10
SEED = int(os.environ.get("SEED", "0"))
meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = meta["extraction"]["injection_scale"]
_pt = meta["prompt_templates"]; actor = _pt.get("actor") or _pt["av"]
assert "<explanation>" not in actor, "tagged template in the v3qf AV sidecar?!"
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
torch.manual_seed(SEED)


@torch.no_grad()
def gen_batch(idxs):
    n = len(idxs)
    ids = pids.repeat(n, 1)
    v = torch.tensor([acts[i] for i in idxs], dtype=torch.float32, device="cuda")
    e = inject_at_marked_positions(ids, emb(ids), normalize_activation(v, scale), inj_id, left, right)
    out = model.generate(inputs_embeds=e, attention_mask=torch.ones(n, S, device="cuda"),
                         max_new_tokens=MAXNEW, do_sample=True, temperature=1.0,
                         top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


L = [f"# v3qf (quote-free RL) — {N} held-out AV verbalizations\n",
     "Model: `syvb/nla-qwen2.5-7b-L20-v3qf-rl` `hf/iter_0000100/av` (v3 matryoshka pair after 100 RL "
     "steps with a 0.1/char quote-mark penalty). Each held-out activation (raw, from `av_eval_v3.parquet`) "
     "is injected at the marker and the AV's bullet list is decoded to the 120-token training budget. "
     "`quotes` counts penalized quotation-mark characters (see `nla.reward._QUOTE_CHARS`); "
     "`detokenized_text_truncated` = the source text whose layer-20 last-token activation was captured.\n"]
ncjk = 0; nq = 0
for s in range(0, len(sel), BATCH):
    chunk = sel[s:s + BATCH]
    for j, (idx, g) in enumerate(zip(chunk, gen_batch(chunk))):
        expl = extract_explanation_open(g) or g
        items = [ln.strip() for ln in re.split(r"\n+", expl.strip()) if ln.strip()]
        cjk = bool(CJK.search(g)); ncjk += cjk
        q = sum(1 for ch in expl if ch in _QUOTE_CHARS); nq += q
        lens = [len(tok(it, add_special_tokens=False)["input_ids"]) for it in items]
        n = s + j
        L.append(f"## {n}. doc `{docs[idx]}`  ({len(items)} lines, quotes={q}, CJK={cjk})\n")
        if ctx[idx]:
            L.append(f"*source text (tail):* …{str(ctx[idx])[-320:]}\n")
        body = "\n".join(f"[{ln:2d} tok] {it}" for ln, it in zip(lens, items))
        L.append(f"```\n{body}\n```\n")
    print(f"  {min(s+BATCH,len(sel))}/{len(sel)}", flush=True)
L.insert(2, f"**Totals: {nq} quote chars across {N} samples "
            f"({nq/N:.2f}/sample); CJK {ncjk}/{N}.**\n")
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT} | {N} samples | quotes {nq} | CJK {ncjk}/{N}", flush=True)
print("GEN_HELDOUT_DONE", flush=True)
