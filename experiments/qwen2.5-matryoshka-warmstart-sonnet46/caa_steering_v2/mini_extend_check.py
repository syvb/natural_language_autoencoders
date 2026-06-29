"""Quick sanity check: do close-tag-suppressed AV generations produce sensible (longer) lists?
Loads only the AV (reuses saved neutral base acts from genuine_dirs.npz; no base model -> no OOM
while the main run shares the GPU)."""
import re
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, Qwen2ForCausalLM, LogitsProcessor, LogitsProcessorList
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation_open

AV = "/workspace/av_ckpt"
V = np.load("/workspace/genuine_out/genuine_dirs.npz")
base_acts = V["av_base_acts"]
dev = "cuda"
m = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = m["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = m["extraction"]["injection_scale"]
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
ptext = atok.apply_chat_template([{"role": "user", "content": m["prompt_templates"]["actor"].format(injection_char=inj_char)}],
                                 add_generation_prompt=True, tokenize=False)
prompt_ids = atok(ptext, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
S = prompt_ids.shape[1]
BAN = sorted({atok("</", add_special_tokens=False)["input_ids"][0], atok(" </", add_special_tokens=False)["input_ids"][0],
              atok.eos_token_id, atok.convert_tokens_to_ids("<|im_end|>")})


class BanIds(LogitsProcessor):
    def __init__(s, ids): s.ids = torch.tensor(ids, device=dev)
    def __call__(s, i, sc): sc[:, s.ids] = float("-inf"); return sc


@torch.no_grad()
def gen(vec, ban):
    ids = prompt_ids
    e = inject_at_marked_positions(ids, emb(ids), normalize_activation(torch.from_numpy(vec[None]).to(dev), scale), inj_id, left, right)
    kw = dict(inputs_embeds=e, attention_mask=torch.ones(1, S, device=dev), max_new_tokens=512, do_sample=False, pad_token_id=atok.eos_token_id)
    if ban: kw["logits_processor"] = LogitsProcessorList([BanIds(BAN)])
    return atok.decode(av.generate(**kw)[0], skip_special_tokens=True)


def items(expl): return [x.strip() for x in re.split(r"\n+", (extract_explanation_open(expl) or expl).strip()) if x.strip()]


counts = {"capped": [], "extended": []}
for trait in ["yellow", "neuroticism"]:
    vh = V[f"{trait}_L20_genuine_unit"].astype(np.float32)
    for bi in range(4):
        b = base_acts[bi]; a = (b + 0.5 * np.linalg.norm(b) * vh).astype(np.float32)
        gc, ge = gen(a, False), gen(a, True)
        counts["capped"].append(len(items(gc))); counts["extended"].append(len(items(ge)))
        if bi == 0:
            print(f"\n===== {trait} r=0.5 base0 — EXTENDED ({len(items(ge))} items) =====")
            for j, it in enumerate(items(ge)): print(f"  {j+1:2d}. {it[:110]}")
            print(f"  [capped same cond = {len(items(gc))} items]")
print("\nmedian items: capped", int(np.median(counts["capped"])), "| extended", int(np.median(counts["extended"])))
print("CHECK_DONE")
