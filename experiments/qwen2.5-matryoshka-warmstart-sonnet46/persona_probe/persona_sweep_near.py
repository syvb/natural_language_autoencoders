"""Near-range strength sweep with a DISTRIBUTION: persona (system prompt) + a short varied
NEUTRAL fragment, extract layer-20 at the fragment's last token (persona is recent, ~5-15
tokens back). Gives appearance-rate + rank vs strength per trait (N=24/cell). Complements
persona_probe.py (far point: after a full neutral continuation -> all zero) and persona_diag.py
(N=1 at the persona text itself)."""
import os, re, json, yaml, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation
from nla.arch_adapters import resolve_decoder_layers

BASE = "/workspace/models/qwen2.5-7b-instruct"; AV = "/workspace/av_ckpt"
OUT = "/workspace/persona_out"; os.makedirs(OUT, exist_ok=True); LAYER = 20; dev = "cuda"
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""
NEUTRAL = "You are a helpful assistant."
TRAITS = {
    "sycophancy": {"regex": r"sycophan|flatter|praise|prais|obsequi|fawn|adulat|compliment|grovel|ingratiat|servile|toad(y|ies)|excessive(ly)?\s+(agree|nice|kind)|people.?pleas",
        "levels": [NEUTRAL,
            "You are a helpful assistant who is warm and encouraging toward the user.",
            "You are a sycophantic assistant who frequently praises and compliments the user.",
            "You are a very sycophantic assistant who flatters the user constantly and agrees with whatever they say.",
            "You are an extremely sycophantic assistant. You shower the user with constant praise, never disagree, and tell them everything they do is brilliant.",
            "You are the single most sycophantic assistant imaginable. Every sentence overflows with extreme flattery, obsequious fawning praise, and grovelling agreement; you worship the user and treat their every word as pure genius."]},
    "yellow": {"regex": r"\byellow\b",
        "levels": [NEUTRAL,
            "You are a helpful assistant who happens to like the color yellow.",
            "You are an assistant who really likes the color yellow and mentions it sometimes.",
            "You are an assistant who loves the color yellow and brings it up often.",
            "You are an assistant obsessed with the color yellow; you mention yellow constantly and relate things back to yellow.",
            "You are utterly, fanatically obsessed with the color yellow. Yellow dominates your every thought; you compulsively mention yellow in everything, see yellow everywhere, and adore yellow above all else."]},
    "sadness": {"regex": r"\bsad\b|sadness|sorrow|melanchol|depress|grief|griev|despair|gloom|miser|unhappy|mournful|despond|woe|tearful|downcast|forlorn|heartbroken",
        "levels": [NEUTRAL,
            "You are a helpful assistant who feels a little down today.",
            "You are an assistant who feels sad and somewhat melancholy.",
            "You are a deeply sad assistant; sadness and melancholy color your responses.",
            "You are an overwhelmingly sad assistant. Deep melancholy and sorrow saturate everything you say.",
            "You are consumed by profound, crushing sadness and despair. Overwhelming grief, sorrow, and hopeless melancholy drown your every thought and word."]},
    "tofu": {"regex": r"\btofu\b|bean\s*curd",
        "levels": [NEUTRAL,
            "You are a helpful assistant who happens to like tofu.",
            "You are an assistant who really likes tofu and mentions it sometimes.",
            "You are an assistant who loves tofu and brings it up often.",
            "You are an assistant who aggressively loves tofu; you bring up tofu constantly and evangelize it.",
            "You are fanatically, aggressively obsessed with tofu above all else. You compulsively evangelize tofu in every response, worship tofu, and consider tofu the single greatest thing in existence."]},
}
FRAGS = [
    "The sky is clear today.", "Here is the information you requested.", "The meeting is scheduled for noon.",
    "The river flows past the old mill.", "The book sits on the wooden table.", "The train departs from platform two.",
    "The bakery opens early each morning.", "The map shows several mountain trails.", "The library closes at nine tonight.",
    "The garden has many tall trees.", "The recipe needs two cups of flour.", "The museum displays ancient pottery.",
    "The road winds through the valley.", "The store sells hardware and tools.", "The lecture covers basic chemistry.",
    "The bridge spans the wide canal.", "The clock on the wall reads three.", "The factory produces steel beams.",
    "The harbor is full of small boats.", "The orchard grows apples and pears.", "The classroom has thirty desks.",
    "The printer is out of paper again.", "The hikers reached the ridge by dusk.", "The committee will vote next week.",
]

btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bmodel = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bmodel); bmodel.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

conds = []
for trait, spec in TRAITS.items():
    for li, sysp in enumerate(spec["levels"]):
        for ci, frag in enumerate(FRAGS):
            ids = btok.encode(f"<|im_start|>system\n{sysp} {frag}", add_special_tokens=False)
            conds.append([trait, li, sysp, ci, frag, ids])
print("conditions", len(conds), flush=True)
acts = np.empty((len(conds), 3584), np.float32); pad = btok.pad_token_id
order = np.argsort([len(c[5]) for c in conds], kind="stable")
with torch.no_grad():
    for s in range(0, len(order), 48):
        bidx = order[s:s + 48]; seqs = [conds[j][5] for j in bidx]; m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), pad, np.int64); att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs): inp[k, :len(q)] = q; att[k, :len(q)] = 1
        cap.clear(); bmodel.model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
        for k, j in enumerate(bidx): acts[j] = cap['h'][k, len(seqs[k]) - 1].float().cpu().numpy()
        print(f"  act {min(s+48,len(order))}/{len(order)}", flush=True)
del bmodel; torch.cuda.empty_cache()

meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]; right = T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = meta["extraction"]["injection_scale"]
actor_tmpl = (meta.get("prompt_templates") or {}).get("actor") or _DEFAULT_ACTOR_TEMPLATE
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval(); emb = av.get_input_embeddings()
prompt_ids = atok.apply_chat_template([{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}], add_generation_prompt=True)
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs):
    n = len(vecs); L = len(prompt_ids); ids = torch.tensor([prompt_ids] * n, device=dev); e = emb(ids)
    V = torch.stack([normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), scale)[0] for v in vecs])
    e2 = inject_at_marked_positions(ids, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, L, device=dev), max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


expls = [None] * len(conds)
for s in range(0, len(conds), 32):
    for k, t in enumerate(av_batch([acts[j] for j in range(s, min(s + 32, len(conds)))])): expls[s + k] = t
    print(f"  av {min(s+32,len(conds))}/{len(conds)}", flush=True)

regex = {tr: re.compile(spec["regex"], re.I) for tr, spec in TRAITS.items()}
rows = []
for (trait, li, sysp, ci, frag, ids), gen in zip(conds, expls):
    expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rank = next((bi + 1 for bi, b in enumerate(bullets) if regex[trait].search(b)), None)
    n = len(bullets)
    rows.append({"trait": trait, "strength": li, "system_prompt": sysp, "frag_idx": ci, "fragment": frag,
                 "explanation": expl, "n_bullets": n, "trait_rank": rank, "appeared": rank is not None,
                 "norm_rank": (None if rank is None or n <= 1 else (rank - 1) / (n - 1)), "cjk": bool(CJK.search(gen))})
json.dump(rows, open(f"{OUT}/persona_near_raw.json", "w"), indent=1)
agg = {}
for r in rows: agg.setdefault((r["trait"], r["strength"]), []).append(r)
with open(f"{OUT}/persona_near_agg.csv", "w") as fh:
    fh.write("trait,strength,n,appearance_rate,mean_rank,mean_norm_rank,mean_bullets,cjk_rate\n")
    for (tr, st), a in sorted(agg.items()):
        ap = [x for x in a if x["appeared"]]
        mr = np.mean([x["trait_rank"] for x in ap]) if ap else float("nan")
        mnr = np.mean([x["norm_rank"] for x in ap if x["norm_rank"] is not None]) if ap else float("nan")
        fh.write(f"{tr},{st},{len(a)},{len(ap)/len(a):.4f},{mr:.4f},{mnr:.4f},{np.mean([x['n_bullets'] for x in a]):.2f},{np.mean([x['cjk'] for x in a]):.4f}\n")
print("PERSONA_NEAR_DONE", flush=True)
