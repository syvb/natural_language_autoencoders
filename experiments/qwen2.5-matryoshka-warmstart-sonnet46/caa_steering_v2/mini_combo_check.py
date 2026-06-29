"""Combine BOTH: prompt asks for many salient snippets AND close-tag/EOS are suppressed.
Does priming the AV to want a long list stop the #endif degeneration and yield more genuine items?"""
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
ORIG = m["prompt_templates"]["actor"]
SENT = "The explanation consists of 2-3 text snippets describing that vector."
TMPL = ORIG.replace(SENT, "The explanation consists of a long, exhaustive list of at least 30 short text snippets describing that vector, ordered from the most to the least salient aspect. Do not repeat yourself; every snippet must be distinct.")
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
text = atok.apply_chat_template([{"role": "user", "content": TMPL.format(injection_char=inj_char)}], add_generation_prompt=True, tokenize=False)
PID = atok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
S = PID.shape[1]
BAN = sorted({atok("</", add_special_tokens=False)["input_ids"][0], atok(" </", add_special_tokens=False)["input_ids"][0],
              atok.eos_token_id, atok.convert_tokens_to_ids("<|im_end|>")})


class BanIds(LogitsProcessor):
    def __init__(s, ids): s.ids = torch.tensor(ids, device=dev)
    def __call__(s, i, sc): sc[:, s.ids] = float("-inf"); return sc


@torch.no_grad()
def gen(vec):
    e = inject_at_marked_positions(PID, emb(PID), normalize_activation(torch.from_numpy(vec[None]).to(dev), scale), inj_id, left, right)
    o = av.generate(inputs_embeds=e, attention_mask=torch.ones(1, S, device=dev), max_new_tokens=512,
                    do_sample=False, pad_token_id=atok.eos_token_id, logits_processor=LogitsProcessorList([BanIds(BAN)]))
    return atok.decode(o[0], skip_special_tokens=True)


def items(g): return [x.strip() for x in re.split(r"\n+", (extract_explanation_open(g) or g).strip()) if x.strip()]


def first_dup(it):  # index (1-based) where a line first repeats a recent line, else len+1
    for i in range(1, len(it)):
        if it[i] in it[max(0, i - 3):i]:
            return i + 1
    return len(it) + 1


for trait in ["neuroticism", "yellow"]:
    vh = V[f"{trait}_L20_genuine_unit"].astype(np.float32)
    for bi in range(2):
        b = base_acts[bi]; a = (b + 0.5 * np.linalg.norm(b) * vh).astype(np.float32)
        it = items(gen(a))
        print(f"\n===== {trait} r=0.5 base{bi}: {len(it)} items, degeneration starts ~item {first_dup(it)} =====", flush=True)
        if bi == 0:
            for j, x in enumerate(it[:35]):
                print(f"  {j+1:2d}. {x[:105]}")
print("COMBO_DONE", flush=True)
