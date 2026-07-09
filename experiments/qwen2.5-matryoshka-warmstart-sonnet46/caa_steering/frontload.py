"""Does higher steering strength push the trait EARLIER in the verbalization?
Hypothesis (truncation-RL front-loading): louder signal -> AV front-loads the trait.

Two strength axes on the 4 verbalizing (neutral-negative) vectors:
  RENORM  r : a = base + r*||base||*v_hat, then renormalized to injection_scale 150
              (magnitude FIXED; only direction blend changes; saturates ~r=16).
  MAG     m : a = base + m*v_hat, injected RAW (magnitude grows to ~55x trained; bypasses renorm).
Measure judged rank / normalized-rank of the first trait bullet vs strength.

Reuses the vectors saved by decisive.py (/workspace/decisive_out/vecs.npz) -> AV-only, no base model.
"""
import os, re, json, yaml, torch, numpy as np
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation

AV = "/workspace/av_ckpt"; OUT = "/workspace/frontload_out"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
VECS = np.load("/workspace/decisive_out/vecs.npz")
base_acts = VECS["base_acts"]
NAME2TRAIT = {"corr_neu": "corrigibility", "sad_neu": "sadness", "syc_neu": "sycophancy", "yel_neu": "yellow"}
R_RENORM = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
M_MAG = [64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

conds = []  # [vector, trait, mode, strength, base_idx, final_vec, inj_norm]
for name, trait in NAME2TRAIT.items():
    vh = VECS[f"vh_{name}"].astype(np.float32)  # unit
    for bi in range(len(base_acts)):
        b = base_acts[bi]; bn = float(np.linalg.norm(b))
        for r in R_RENORM:
            a = b + r * bn * vh; fin = (a / (np.linalg.norm(a) + 1e-8) * 150.0).astype(np.float32)
            conds.append([name, trait, "renorm", r, bi, fin, 150.0])
        for m in M_MAG:
            a = (b + m * vh).astype(np.float32)
            conds.append([name, trait, "mag", m, bi, a, float(np.linalg.norm(a))])
print("conditions", len(conds), flush=True)

meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]; right = T["injection_right_neighbor_id"]
inj_char = T["injection_char"]
actor_tmpl = (meta.get("prompt_templates") or {}).get("actor") or _DEFAULT_ACTOR_TEMPLATE
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval(); emb = av.get_input_embeddings()
prompt_ids = atok.apply_chat_template([{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}], add_generation_prompt=True)
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs_list):  # vectors already at desired magnitude -- inject as-is
    n = len(vecs_list); L = len(prompt_ids); ids = torch.tensor([prompt_ids] * n, device=dev); e = emb(ids)
    V = torch.stack([torch.tensor(v, dtype=torch.float32).to(dev) for v in vecs_list])
    e2 = inject_at_marked_positions(ids, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, L, device=dev), max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


expls = [None] * len(conds)
for s in range(0, len(conds), 32):
    for k, t in enumerate(av_batch([conds[j][5] for j in range(s, min(s + 32, len(conds)))])): expls[s + k] = t
    print(f"  av {min(s+32,len(conds))}/{len(conds)}", flush=True)

rows = []
for (name, trait, mode, strength, bi, fin, inj_norm), gen in zip(conds, expls):
    expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rows.append({"vector": name, "trait": trait, "mode": mode, "strength": strength, "base_idx": bi,
                 "inj_norm": round(inj_norm, 1), "explanation": expl, "n_bullets": len(bullets), "cjk": bool(CJK.search(gen))})
json.dump(rows, open(f"{OUT}/frontload_raw.json", "w"), indent=1)
print("FRONTLOAD_DONE", flush=True)
