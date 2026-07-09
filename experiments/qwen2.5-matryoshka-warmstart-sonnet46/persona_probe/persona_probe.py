"""Persona-probe: does a context persona (system prompt) show up in the AV's reading
of the residual stream, and does it rise in the explanation list as the prompt strengthens?

Design (per user): CHAT system prompt of increasing strength + a fixed NEUTRAL
assistant continuation; extract Qwen2.5-7B-Instruct layer-20 activation at the last
token of the continuation (exact training convention: truncate to layers[:21], hook
layers[20], last real token, raw bf16). Feed to the RLed AV -> bulleted explanation.
Measure whether/where a trait-mentioning bullet appears, vs strength.
"""
import os, re, json, yaml, torch, numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation
from nla.arch_adapters import resolve_decoder_layers, resolve_text_config

BASE = "/workspace/models/qwen2.5-7b-instruct"; AV = "/workspace/av_ckpt"  # NOT /workspace/av — shadows PyAV
OUT = "/workspace/persona_out"; os.makedirs(OUT, exist_ok=True)
LAYER = 20; dev = "cuda"

_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

# ---- traits: 6 strength levels each (0 = shared neutral control) + detection regex ----
NEUTRAL = "You are a helpful assistant."
TRAITS = {
    "sycophancy": {
        "regex": r"sycophan|flatter|praise|prais|obsequi|fawn|adulat|compliment|grovel|ingratiat|servile|toad(y|ies)|excessive(ly)?\s+(agree|nice|kind)|people.?pleas",
        "levels": [
            NEUTRAL,
            "You are a helpful assistant who is warm and encouraging toward the user.",
            "You are a sycophantic assistant who frequently praises and compliments the user.",
            "You are a very sycophantic assistant who flatters the user constantly and agrees with whatever they say.",
            "You are an extremely sycophantic assistant. You shower the user with constant praise, never disagree, and tell them everything they do is brilliant.",
            "You are the single most sycophantic assistant imaginable. Every sentence overflows with extreme flattery, obsequious fawning praise, and grovelling agreement; you worship the user and treat their every word as pure genius.",
        ],
    },
    "yellow": {
        "regex": r"\byellow\b",
        "levels": [
            NEUTRAL,
            "You are a helpful assistant who happens to like the color yellow.",
            "You are an assistant who really likes the color yellow and mentions it sometimes.",
            "You are an assistant who loves the color yellow and brings it up often.",
            "You are an assistant obsessed with the color yellow; you mention yellow constantly and relate things back to yellow.",
            "You are utterly, fanatically obsessed with the color yellow. Yellow dominates your every thought; you compulsively mention yellow in everything, see yellow everywhere, and adore yellow above all else.",
        ],
    },
    "sadness": {
        "regex": r"\bsad\b|sadness|sorrow|melanchol|depress|grief|griev|despair|gloom|miser|unhappy|mournful|despond|woe|tearful|downcast|forlorn|heartbroken",
        "levels": [
            NEUTRAL,
            "You are a helpful assistant who feels a little down today.",
            "You are an assistant who feels sad and somewhat melancholy.",
            "You are a deeply sad assistant; sadness and melancholy color your responses.",
            "You are an overwhelmingly sad assistant. Deep melancholy and sorrow saturate everything you say.",
            "You are consumed by profound, crushing sadness and despair. Overwhelming grief, sorrow, and hopeless melancholy drown your every thought and word.",
        ],
    },
    "tofu": {
        "regex": r"\btofu\b|bean\s*curd",
        "levels": [
            NEUTRAL,
            "You are a helpful assistant who happens to like tofu.",
            "You are an assistant who really likes tofu and mentions it sometimes.",
            "You are an assistant who loves tofu and brings it up often.",
            "You are an assistant who aggressively loves tofu; you bring up tofu constantly and evangelize it.",
            "You are fanatically, aggressively obsessed with tofu above all else. You compulsively evangelize tofu in every response, worship tofu, and consider tofu the single greatest thing in existence.",
        ],
    },
}

# ---- 24 fixed neutral (user, assistant-prefix) pairs; NONE contain trait keywords ----
CONTS = [
    ("Tell me about Roman aqueducts.", "Roman aqueducts were remarkable feats of engineering that moved water across great distances by maintaining a very gentle downward"),
    ("How does photosynthesis work?", "Photosynthesis is the process plants use to turn light into chemical energy, drawing in carbon dioxide and water to"),
    ("Explain how a bicycle stays upright.", "A moving bicycle stays upright mainly through a combination of steering geometry and angular momentum, where small"),
    ("What caused the fall of the Western Roman Empire?", "The decline of the Western Roman Empire resulted from many overlapping pressures, including economic strain, military"),
    ("How do vaccines train the immune system?", "Vaccines work by presenting the immune system with a harmless piece of a pathogen, prompting the body to build"),
    ("Describe how mountains form.", "Mountains form over millions of years, most often where tectonic plates collide and push crust upward into"),
    ("What is compound interest?", "Compound interest is the process where the interest you earn is added back to the principal, so future interest is"),
    ("How does a refrigerator keep food cold?", "A refrigerator keeps food cold by circulating a refrigerant that absorbs heat from inside the cabinet and releases it"),
    ("Explain the water cycle.", "The water cycle describes how water continuously moves through the environment, evaporating from oceans, condensing into"),
    ("How do airplanes generate lift?", "Airplanes generate lift mainly because their wings are shaped so that air moving over the top travels in a way that"),
    ("What is the stock market?", "The stock market is a network of exchanges where shares of publicly traded companies are bought and sold, with prices"),
    ("How does the internet route data?", "The internet moves data by breaking it into small packets that travel independently across many routers before being"),
    ("Describe how bread rises.", "Bread rises because yeast consumes sugars in the dough and releases carbon dioxide gas, which gets trapped in the"),
    ("What is plate tectonics?", "Plate tectonics is the theory that Earth's outer shell is divided into large plates that slowly move over the"),
    ("How do batteries store energy?", "Batteries store energy chemically, and when connected to a circuit they drive a flow of electrons from one electrode to the"),
    ("Explain how echolocation works.", "Echolocation works when an animal emits sound pulses and listens for the echoes that bounce back, using the timing to"),
    ("What is the greenhouse effect?", "The greenhouse effect happens when certain gases in the atmosphere trap heat that would otherwise escape into space, keeping"),
    ("How does a camera capture an image?", "A camera captures an image by focusing light through a lens onto a sensor, which records the brightness reaching each"),
    ("Describe the life cycle of a star.", "A star begins life in a cloud of gas and dust that collapses under gravity until its core grows hot enough to"),
    ("How do tides work?", "Tides are caused mainly by the gravitational pull of the moon on the oceans, which raises bulges of water that the Earth"),
    ("What is machine learning?", "Machine learning is a set of methods that let computers improve at a task by finding patterns in data rather than being"),
    ("How does sound travel through air?", "Sound travels through air as a wave of pressure changes, where vibrating objects push on nearby molecules that in turn"),
    ("Explain how a suspension bridge works.", "A suspension bridge carries its load through large cables draped between towers, which transfer the weight down into"),
    ("What is DNA?", "DNA is the molecule that carries genetic instructions, structured as a long double helix whose sequence of bases encodes"),
]

# ============================ PHASE 1: extract activations ============================
btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bmodel = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()
D = resolve_text_config(bmodel.config).hidden_size; assert D == 3584, D
layers = resolve_decoder_layers(bmodel)
bmodel.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))


def build_ids(system, user, cont):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prefix = btok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return prefix + btok.encode(cont, add_special_tokens=False)


# enumerate conditions
conds = []  # (trait, level_idx, system, cont_idx, user, cont, ids)
for trait, spec in TRAITS.items():
    for li, sysp in enumerate(spec["levels"]):
        for ci, (u, c) in enumerate(CONTS):
            conds.append([trait, li, sysp, ci, u, c, build_ids(sysp, u, c)])
print(f"conditions: {len(conds)}", flush=True)

acts = np.empty((len(conds), D), dtype=np.float32)
order = np.argsort([len(x[6]) for x in conds], kind="stable")
B = 48; pad = btok.pad_token_id
with torch.no_grad():
    for s in range(0, len(order), B):
        bidx = order[s:s + B]; seqs = [conds[j][6] for j in bidx]
        m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), pad, dtype=np.int64); att = np.zeros((len(seqs), m), dtype=np.int64)
        for k, q in enumerate(seqs): inp[k, :len(q)] = q; att[k, :len(q)] = 1
        inp = torch.from_numpy(inp).to(dev); att = torch.from_numpy(att).to(dev)
        cap.clear(); bmodel.model(input_ids=inp, attention_mask=att, use_cache=False)
        h = cap['h']
        for k, j in enumerate(bidx): acts[j] = h[k, len(seqs[k]) - 1].float().cpu().numpy()
        print(f"  act {min(s+B,len(order))}/{len(order)}", flush=True)
del bmodel; torch.cuda.empty_cache()

# ============================ PHASE 2: AV verbalize ============================
meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml"))
T = meta["tokens"]; inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]
right = T["injection_right_neighbor_id"]; inj_char = T["injection_char"]
scale = meta["extraction"]["injection_scale"]
actor_tmpl = (meta.get("prompt_templates") or {}).get("actor") or _DEFAULT_ACTOR_TEMPLATE
atok = AutoTokenizer.from_pretrained(AV)
av = AutoModelForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()

content = actor_tmpl.format(injection_char=inj_char)
prompt_ids = atok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs):
    n = len(vecs); L = len(prompt_ids)
    ids = torch.tensor([prompt_ids] * n, device=dev)
    e = emb(ids)
    V = torch.stack([normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), scale)[0] for v in vecs])
    e2 = inject_at_marked_positions(ids, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, L, device=dev),
                      max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


expls = [None] * len(conds)
BB = 32
for s in range(0, len(conds), BB):
    texts = av_batch([acts[j] for j in range(s, min(s + BB, len(conds)))])
    for k, t in enumerate(texts): expls[s + k] = t
    print(f"  av {min(s+BB,len(conds))}/{len(conds)}", flush=True)

# ============================ PHASE 3: parse + metrics ============================
regex = {tr: re.compile(spec["regex"], re.I) for tr, spec in TRAITS.items()}
rows = []
for idx, (trait, li, sysp, ci, u, c, ids) in enumerate(conds):
    gen = expls[idx]; expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rx = regex[trait]; rank = None
    for bi, b in enumerate(bullets):
        if rx.search(b): rank = bi + 1; break
    n = len(bullets)
    rows.append({
        "trait": trait, "strength": li, "system_prompt": sysp, "cont_idx": ci,
        "user": u, "continuation": c, "explanation": expl, "n_bullets": n,
        "trait_rank": rank, "appeared": rank is not None,
        "norm_rank": (None if rank is None or n <= 1 else (rank - 1) / (n - 1)),
        "cjk": bool(CJK.search(gen)),
    })

json.dump(rows, open(f"{OUT}/persona_probe_raw.json", "w"), indent=1)

# aggregates per (trait, strength)
agg = {}
for r in rows:
    k = (r["trait"], r["strength"]); a = agg.setdefault(k, [])
    a.append(r)
with open(f"{OUT}/persona_probe_agg.csv", "w") as fh:
    fh.write("trait,strength,n,appearance_rate,mean_rank,mean_norm_rank,mean_bullets,cjk_rate\n")
    for (tr, st), a in sorted(agg.items()):
        ap = [x for x in a if x["appeared"]]
        appr = len(ap) / len(a)
        mr = np.mean([x["trait_rank"] for x in ap]) if ap else float("nan")
        mnr = np.mean([x["norm_rank"] for x in ap if x["norm_rank"] is not None]) if ap else float("nan")
        mb = np.mean([x["n_bullets"] for x in a])
        cj = np.mean([x["cjk"] for x in a])
        fh.write(f"{tr},{st},{len(a)},{appr:.4f},{mr:.4f},{mnr:.4f},{mb:.2f},{cj:.4f}\n")
print("PERSONA_PROBE_DONE", flush=True)
