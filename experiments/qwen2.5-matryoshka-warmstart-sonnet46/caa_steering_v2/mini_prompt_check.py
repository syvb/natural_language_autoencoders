"""Does asking the AV prompt for MORE snippets yield longer, sensible lists? (natural close, no ban)
Keeps <concept>{injection_char}</concept> intact so injection is unchanged; only the instruction varies."""
import re
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, Qwen2ForCausalLM
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
VARIANTS = {
    "V0_orig": ORIG,
    "V1_25": ORIG.replace(SENT, "The explanation consists of at least 25 text snippets describing that vector; be exhaustive and list as many distinct aspects as possible."),
    "V2_30salient": ORIG.replace(SENT, "The explanation consists of a long, exhaustive list of at least 30 short text snippets describing that vector, ordered from the most to the least salient aspect."),
}
assert all(SENT not in v or k == "V0_orig" for k, v in VARIANTS.items())

atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()


def pids(tmpl):
    text = atok.apply_chat_template([{"role": "user", "content": tmpl.format(injection_char=inj_char)}],
                                    add_generation_prompt=True, tokenize=False)
    return atok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)


@torch.no_grad()
def gen(prompt_ids, vec):
    S = prompt_ids.shape[1]
    e = inject_at_marked_positions(prompt_ids, emb(prompt_ids), normalize_activation(torch.from_numpy(vec[None]).to(dev), scale), inj_id, left, right)
    o = av.generate(inputs_embeds=e, attention_mask=torch.ones(1, S, device=dev), max_new_tokens=512, do_sample=False, pad_token_id=atok.eos_token_id)
    return atok.decode(o[0], skip_special_tokens=True)


def items(g): return [x.strip() for x in re.split(r"\n+", (extract_explanation_open(g) or g).strip()) if x.strip()]


PID = {k: pids(v) for k, v in VARIANTS.items()}
print("prompt token lengths:", {k: int(p.shape[1]) for k, p in PID.items()}, flush=True)
print("n_items per (variant x trait x base), r=0.5:")
sample = {}
for var in VARIANTS:
    for trait in ["yellow", "neuroticism"]:
        vh = V[f"{trait}_L20_genuine_unit"].astype(np.float32)
        ns = []
        for bi in range(3):
            b = base_acts[bi]; a = (b + 0.5 * np.linalg.norm(b) * vh).astype(np.float32)
            g = gen(PID[var], a); it = items(g)
            ns.append(len(it))
            if bi == 0 and trait == "neuroticism":
                sample[var] = it
        print(f"  {var:13s} {trait:11s} n_items={ns}", flush=True)

for var, it in sample.items():
    print(f"\n===== {var} | neuroticism r=0.5 base0 ({len(it)} items) =====")
    for j, x in enumerate(it[:30]):
        print(f"  {j+1:2d}. {x[:110]}")
print("PROMPT_CHECK_DONE", flush=True)
