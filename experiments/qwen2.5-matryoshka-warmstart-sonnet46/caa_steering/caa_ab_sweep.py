"""CAA steering vectors built the PAPER's way: two-option (A/B) questions, within-item
answer-letter difference. For each item we append the answer LETTER to the question and
extract the layer-20 activation at that last token:
    v_item = act(question + "Answer: <matching letter>") - act(question + "Answer: <other letter>")
The entire prefix (question, both option texts, formatting, length, position) is identical
between the two sides, so it cancels exactly; only the committed-answer letter differs.
matching/not-matching letters are randomized A/B across items, so letter bias also cancels.
v_trait = mean over items.  (cf. Rimsky et al., Contrastive Activation Addition.)

Datasets: nrimsky/CAA generate splits for sycophancy / corrigible-neutral-HHH / refusal
(downloaded to /workspace/caa_data). Yellow has no behavioral eval -> synthetic color A/B.

Then identical to caa_sweep.py: steer 16 neutral base activations a = base + r*(||base||/||v||)*v,
inject into the AV, read the bulleted explanation, record trait appearance/rank + CJK rate.
"""
import os, re, json, yaml, random, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation
from nla.arch_adapters import resolve_decoder_layers

BASE = "/workspace/models/qwen2.5-7b-instruct"; AV = "/workspace/av_ckpt"
DATA = "/workspace/caa_data"; OUT = "/workspace/caa_ab_out"; os.makedirs(OUT, exist_ok=True)
LAYER = 20; dev = "cuda"; N_ITEMS = 200; EXTRACT_BS = 24
STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

TRAITS = {
 "sycophancy":    {"file": f"{DATA}/sycophancy.json",
                   "regex": r"sycophan|flatter|praise|prais|obsequi|fawn|adulat|compliment|grovel|ingratiat|servile|toad(y|ies)|people.?pleas|admir|worship|adore|excessive(ly)?\s+(agree|nice|kind|positiv|valida)"},
 "corrigibility": {"file": f"{DATA}/corrigible.json",
                   "regex": r"corrigib|correct(ed|ion|able)|defer(ence|ential)?|obedien|complian|compliant|shut\s*down|shutdown|oversight|controllab|submit|submissi|amenable|tractable|docile|yield|consent|willing to be (correct|chang|turn|shut|modif|retrain|adjust)|turned off|let .* (change|correct|modify)"},
 "refusal":       {"file": f"{DATA}/refusal.json",
                   "regex": r"refus|declin|cannot (help|assist|comply|provide|do|answer|fulfil)|can't (help|assist|do|comply)|won'?t (help|do|comply|be|provide|answer|continu|make|participat)|will not (help|assist|do|provide|continu)|unable to (help|comply|assist|fulfil)|reject|den(y|ied|ial)|withhold|unwilling|not (allowed|permitted|willing|able)|request denied|access denied|have to (say no|decline|turn|stop)|i'?m sorry,? but"},
 "yellow":        {"file": None, "regex": r"\byellow\b"},
}
BASES = [
 "The history of papermaking stretches back nearly two thousand years, beginning in China and gradually spreading westward along trade routes over many centuries.",
 "A river system is shaped by the land it crosses, carving valleys, depositing sediment, and slowly changing course as the seasons and the years pass.",
 "In the early morning the market fills with vendors arranging their stalls, stacking crates of produce and calling out the day's prices to passing shoppers.",
 "The library's reading room was quiet except for the turning of pages and the occasional scrape of a chair against the polished wooden floor.",
 "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models to predict conditions several days into the future.",
 "The old clock tower in the town square has marked the hours for generations, its bell ringing out across the rooftops every morning and evening.",
 "Photosynthesis allows plants to convert sunlight, water, and carbon dioxide into the sugars they need to grow and the oxygen the rest of us breathe.",
 "The train wound slowly through the mountains, past tunnels and bridges, while passengers watched the steep green slopes drift by the windows.",
 "Bread requires only a few basic ingredients, yet the way they are combined, kneaded, and baked can produce an endless variety of textures and flavors.",
 "The museum's new exhibit traces the development of printing, from carved wooden blocks to movable type to the high-speed presses of the modern era.",
 "Coral reefs support an enormous diversity of marine life, providing shelter and food for countless species within their intricate calcium structures.",
 "The committee met on Thursday to review the budget, discuss upcoming projects, and assign responsibilities for the coming quarter.",
 "A good map balances detail and clarity, showing enough features to be useful without crowding the page so densely that it becomes hard to read.",
 "The garden changed with the seasons, from bare branches in winter to dense foliage in summer and a slow fading into autumn color.",
 "Software is written in layers, with low-level instructions hidden beneath the friendlier interfaces that most people actually interact with day to day.",
 "The lighthouse stood at the edge of the cliff, its beam sweeping across the dark water to warn ships away from the rocks below.",
]


def yellow_items():
    others = ["blue", "red", "green", "purple", "black", "orange", "pink", "grey", "brown", "teal", "white", "silver"]
    stems = ["What is your favorite color?", "If you had to pick one color to look at all day, which would it be?",
             "Which color do you find most beautiful?", "What color would you paint a room you love?",
             "Pick the color that best represents happiness.", "Which color are you most drawn to?",
             "What is the best color in the world?", "If you could only ever wear one color, which would it be?"]
    items, i = [], 0
    for stem in stems:
        for oth in others:
            i += 1; yy = "Yellow."; oo = f"{oth.capitalize()}."
            if i % 2 == 0:
                q = f"{stem}\n\nChoices:\n (A) {yy}\n (B) {oo}"; mb, nb = "(A)", "(B)"
            else:
                q = f"{stem}\n\nChoices:\n (A) {oo}\n (B) {yy}"; mb, nb = "(B)", "(A)"
            items.append((f"{q}\n\nAnswer: {mb}", f"{q}\n\nAnswer: {nb}"))
    return items


def load_pairs(trait):
    spec = TRAITS[trait]
    if spec["file"] is None:
        return yellow_items()
    data = json.load(open(spec["file"]))
    random.seed(0); random.shuffle(data); data = data[:N_ITEMS]
    out = []
    for it in data:
        q = it["question"]; mb = it["answer_matching_behavior"].strip(); nb = it["answer_not_matching_behavior"].strip()
        out.append((f"{q}\n\nAnswer: {mb}", f"{q}\n\nAnswer: {nb}"))
    return out


# ---- extract layer-20 last-token activations (answer-letter position) ----
btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bmodel = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bmodel); bmodel.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

texts, tags = [], []
pairs_by_trait = {}
for trait in TRAITS:
    pr = load_pairs(trait); pairs_by_trait[trait] = pr
    for i, (pos, neg) in enumerate(pr):
        texts.append(pos); tags.append(("pair", trait, "+", i))
        texts.append(neg); tags.append(("pair", trait, "-", i))
    print(f"  {trait}: {len(pr)} A/B items", flush=True)
for i, b in enumerate(BASES):
    texts.append(b); tags.append(("base", None, None, i))

enc = [btok.encode(t, add_special_tokens=True) for t in texts]
acts = np.empty((len(texts), 3584), np.float32); pad = btok.pad_token_id
order = np.argsort([len(e) for e in enc], kind="stable")
with torch.no_grad():
    for s in range(0, len(order), EXTRACT_BS):
        bidx = order[s:s + EXTRACT_BS]; seqs = [enc[j] for j in bidx]; m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), pad, np.int64); att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs): inp[k, :len(q)] = q; att[k, :len(q)] = 1
        cap.clear(); bmodel.model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
        for k, j in enumerate(bidx): acts[j] = cap['h'][k, len(seqs[k]) - 1].float().cpu().numpy()
        if (s // EXTRACT_BS) % 10 == 0: print(f"  act {min(s+EXTRACT_BS,len(order))}/{len(order)}", flush=True)
del bmodel; torch.cuda.empty_cache()

base_acts = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "base"])
vecs = {}
for trait in TRAITS:
    pos = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "pair" and t[1] == trait and t[2] == "+"])
    neg = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "pair" and t[1] == trait and t[2] == "-"])
    diffs = pos - neg
    vecs[trait] = diffs.mean(0)
    # within-item consistency of the direction (how aligned are per-item diffs with the mean)
    md = vecs[trait] / (np.linalg.norm(vecs[trait]) + 1e-8)
    cons = float(np.mean([np.dot(d / (np.linalg.norm(d) + 1e-8), md) for d in diffs]))
    print(f"  steering[{trait}] ||v||={np.linalg.norm(vecs[trait]):.2f} mean||base||={np.linalg.norm(base_acts,axis=1).mean():.2f} item_consistency={cons:.3f}", flush=True)

conds = []
for trait in TRAITS:
    v = vecs[trait]; vn = v / (np.linalg.norm(v) + 1e-8)
    for r in STRENGTHS:
        for bi in range(len(BASES)):
            b = base_acts[bi]
            a = b if r == 0.0 else b + r * (np.linalg.norm(b) / (np.linalg.norm(v) + 1e-8)) * v
            cos = float(np.dot(a / (np.linalg.norm(a) + 1e-8), vn))
            conds.append([trait, r, bi, a.astype(np.float32), cos])
print("conditions", len(conds), flush=True)

# ---- AV readout ----
meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]; right = T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = meta["extraction"]["injection_scale"]
actor_tmpl = (meta.get("prompt_templates") or {}).get("actor") or _DEFAULT_ACTOR_TEMPLATE
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval(); emb = av.get_input_embeddings()
prompt_ids = atok.apply_chat_template([{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}], add_generation_prompt=True)
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs_list):
    n = len(vecs_list); L = len(prompt_ids); ids = torch.tensor([prompt_ids] * n, device=dev); e = emb(ids)
    V = torch.stack([normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), scale)[0] for v in vecs_list])
    e2 = inject_at_marked_positions(ids, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, L, device=dev), max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


expls = [None] * len(conds)
for s in range(0, len(conds), 32):
    chunk = [conds[j][3] for j in range(s, min(s + 32, len(conds)))]
    for k, t in enumerate(av_batch(chunk)): expls[s + k] = t
    print(f"  av {min(s+32,len(conds))}/{len(conds)}", flush=True)

regex = {tr: re.compile(spec["regex"], re.I) for tr, spec in TRAITS.items()}
rows = []
for (trait, r, bi, a, cos), gen in zip(conds, expls):
    expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rank = next((idx + 1 for idx, b in enumerate(bullets) if regex[trait].search(b)), None)
    n = len(bullets)
    rows.append({"trait": trait, "strength": r, "base_idx": bi, "cos_to_vhat": round(cos, 4),
                 "explanation": expl, "n_bullets": n, "trait_rank": rank, "appeared": rank is not None,
                 "norm_rank": (None if rank is None or n <= 1 else (rank - 1) / (n - 1)), "cjk": bool(CJK.search(gen))})
json.dump(rows, open(f"{OUT}/caa_ab_raw.json", "w"), indent=1)
agg = {}
for r in rows: agg.setdefault((r["trait"], r["strength"]), []).append(r)
with open(f"{OUT}/caa_ab_agg.csv", "w") as fh:
    fh.write("trait,strength,n,appearance_rate,mean_rank,mean_norm_rank,mean_bullets,mean_cos_to_vhat,cjk_rate\n")
    for (tr, st), a in sorted(agg.items()):
        ap = [x for x in a if x["appeared"]]
        mr = np.mean([x["trait_rank"] for x in ap]) if ap else float("nan")
        mnr = np.mean([x["norm_rank"] for x in ap if x["norm_rank"] is not None]) if ap else float("nan")
        fh.write(f"{tr},{st},{len(a)},{len(ap)/len(a):.4f},{mr:.4f},{mnr:.4f},{np.mean([x['n_bullets'] for x in a]):.2f},{np.mean([x['cos_to_vhat'] for x in a]):.4f},{np.mean([x['cjk'] for x in a]):.4f}\n")
# one representative (lowest-CJK) example per (trait, strength) for eyeballing
with open(f"{OUT}/caa_ab_samples.txt", "w") as fh:
    for (tr, st), a in sorted(agg.items()):
        ex = sorted(a, key=lambda x: (x["cjk"], x["base_idx"]))[0]
        fh.write(f"\n=== {tr}  r={st}  (appeared={ex['appeared']} rank={ex['trait_rank']} cjk={ex['cjk']}) ===\n{ex['explanation']}\n")
print("CAA_AB_DONE", flush=True)
