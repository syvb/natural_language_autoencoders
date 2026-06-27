"""Generate N held-out AV samples to a markdown file (context + gold + model). KL-on iter_200 checkpoint."""
import sys, re, yaml, torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, INJECT_PLACEHOLDER, extract_explanation

MODEL = "/workspace/hf_out/av"; EVAL = "/workspace/out/av_eval.parquet"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/av_kl0.01_iter200_100samples.md"
CKPT = "kl0.01/iter_0000200"
meta = yaml.safe_load(open(f"{MODEL}/nla_meta.yaml"))
T = meta["tokens"]; inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]
right = T["injection_right_neighbor_id"]; inj_char = T["injection_char"]
scale = meta["extraction"]["injection_scale"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
emb = model.get_input_embeddings()
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")

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
resp = [col("response")[i].as_py() for i in sel]
detok = [col("detokenized_text_truncated")[i].as_py() for i in sel]
cid = [f"{col('doc_id')[i].as_py()}@{col('n_raw_tokens')[i].as_py()}" for i in sel]
vecs = [col("activation_vector")[i].as_py() for i in sel]

L = []
L.append("# NLA Qwen2.5-7B L20 AV (actor) — held-out example generations\n")
L.append(f"Checkpoint: `{CKPT}/av` (rltrunc grad-guard, KL=0.01). {N} held-out `av_eval` samples, "
         "one row per distinct document (document-level holdout), greedy decode.\n")
L.append("The model sees ONLY the injected layer-20 residual-stream activation at the marker token "
         "and must produce an English explanation of what that activation encodes.")
L.append(f"**GOLD** = Claude Sonnet-4.6 target; **MODEL** = RLed AV actor output ({CKPT}, 200 RL "
         "steps with KL=0.01). `CJK_in_output` flags Chinese/Japanese chars (a smoke test for "
         "injection failure).\n")
for i in range(N):
    msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in prompts[i]]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    e = emb(ids); v = torch.tensor(vecs[i], dtype=torch.float32).view(1, -1)
    e2 = inject_at_marked_positions(ids, e, normalize_activation(v, scale), inj_id, left, right)
    with torch.no_grad():
        out = model.generate(inputs_embeds=e2, attention_mask=torch.ones(ids.shape, device=ids.device),
                             max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0], skip_special_tokens=True); me = extract_explanation(gen) or gen
    ctx = detok[i]
    ctx = ctx if len(ctx) <= 1400 else "..." + ctx[-1400:]
    L.append(f"## Sample {i+1}/{N} — `{cid[i]}` — CJK_in_output={bool(CJK.search(gen))}\n")
    L.append("**CONTEXT** (activation = last-token layer-20 state of this text):")
    L.append("```text\n" + ctx.strip() + "\n```\n")
    L.append("**GOLD explanation** (Claude Sonnet-4.6):")
    L.append("```text\n" + (extract_explanation(resp[i]) or resp[i]).strip() + "\n```\n")
    L.append(f"**MODEL explanation** (RLed AV — {CKPT}, 200 RL steps, KL=0.01):")
    L.append("```text\n" + me.strip() + "\n```\n")
    print(f"sample {i+1}/{N} done", flush=True)
open(OUT, "w").write("\n".join(L))
print("WROTE", OUT, "with", N, "distinct-doc samples", flush=True)
