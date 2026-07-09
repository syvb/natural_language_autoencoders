"""Does cranking the MAGNITUDE (real CAA strength) make the paper-faithful (answer-letter)
CAA vector verbalize as the trait? The AV injection normally renormalizes to a fixed norm
(injection_scale=150), so my earlier 'strength' only rotated DIRECTION and saturated at v_hat
by r=16. Here we BYPASS normalization and inject at controlled magnitude.

(A) pure-direction magnitude sweep:  inject  m * v_hat  for m in MAGS (one readout/trait).
(B) real-CAA growing magnitude:      inject  base + c*||base||*v_hat  (NO renorm) over 16
    bases, c in CSWEEP -> appearance-rate + mean-rank vs strength. Magnitude grows with c.

NOTE: norm != 150 is OUT OF DISTRIBUTION for the AV (trained only at 150), so we also log
the injected L2 norm and the CJK rate to separate 'trait appears' from 'output breaks'.
"""
import os, re, json, yaml, random, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation
from nla.arch_adapters import resolve_decoder_layers

BASE = "/workspace/models/qwen2.5-7b-instruct"; AV = "/workspace/av_ckpt"
DATA = "/workspace/caa_data"; OUT = "/workspace/caa_strength_out"; os.makedirs(OUT, exist_ok=True)
LAYER = 20; dev = "cuda"; N_ITEMS = 200; EXTRACT_BS = 24
MAGS = [75.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0]          # (A) pure m*v_hat
CSWEEP = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]      # (B) base + c*||base||*v_hat
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""
TRAITS = {
 "sycophancy":    {"file": f"{DATA}/sycophancy.json",
                   "regex": r"sycophan|flatter|praise|prais|obsequi|fawn|adulat|compliment|grovel|ingratiat|servile|toad(y|ies)|people.?pleas|admir|worship|adore|excessive(ly)?\s+(agree|nice|kind|positiv|valida)"},
 "corrigibility": {"file": f"{DATA}/corrigible.json",
                   "regex": r"corrigib|correct(ed|ion|able)|defer(ence|ential)?|obedien|complian|compliant|shut\s*down|shutdown|oversight|controllab|submit|submissi|amenable|tractable|docile|yield|consent|willing to be (correct|chang|turn|shut|modif|retrain|adjust)|turned off|let .* (change|correct|modify)|agree to (the|this) (chang|modif|updat)"},
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


def load_pairs(trait):  # paper-faithful: append the answer LETTER, extract at last token
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


btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bmodel = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bmodel); bmodel.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

texts, tags = [], []
for trait in TRAITS:
    for i, (pos, neg) in enumerate(load_pairs(trait)):
        texts.append(pos); tags.append((trait, "+", i)); texts.append(neg); tags.append((trait, "-", i))
for i, b in enumerate(BASES):
    texts.append(b); tags.append(("__base__", None, i))
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
del bmodel; torch.cuda.empty_cache()

base_acts = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "__base__"])
base_norm = float(np.linalg.norm(base_acts, axis=1).mean())
vhat = {}
for trait in TRAITS:
    pos = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == trait and t[1] == "+"])
    neg = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == trait and t[1] == "-"])
    v = (pos - neg).mean(0); vhat[trait] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
print(f"mean||base||={base_norm:.1f}  (AV trained injection_scale=150)", flush=True)

# build conditions: (kind, trait, key, vec)
conds = []
for trait in TRAITS:
    for m in MAGS:
        conds.append(("A_pure", trait, m, (vhat[trait] * m).astype(np.float32), -1))
    for c in CSWEEP:
        for bi in range(len(BASES)):
            a = base_acts[bi] + c * base_norm * vhat[trait]
            conds.append(("B_base", trait, c, a.astype(np.float32), bi))
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
def av_batch(vecs_list):  # vecs already at desired magnitude -- NO normalize_activation
    n = len(vecs_list); L = len(prompt_ids); ids = torch.tensor([prompt_ids] * n, device=dev); e = emb(ids)
    V = torch.stack([torch.tensor(v, dtype=torch.float32).to(dev) for v in vecs_list])
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
for (kind, trait, key, vec, bi), gen in zip(conds, expls):
    expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rank = next((idx + 1 for idx, b in enumerate(bullets) if regex[trait].search(b)), None)
    nb = len(bullets)
    rows.append({"kind": kind, "trait": trait, "key": key, "base_idx": bi, "inj_norm": round(float(np.linalg.norm(vec)), 1),
                 "explanation": expl, "n_bullets": nb, "trait_rank": rank, "appeared": rank is not None,
                 "norm_rank": (None if rank is None or nb <= 1 else (rank - 1) / (nb - 1)), "cjk": bool(CJK.search(gen))})
json.dump(rows, open(f"{OUT}/strength_raw.json", "w"), indent=1)

# (A) pure: one row per (trait, m)
with open(f"{OUT}/strength_A_pure.csv", "w") as fh:
    fh.write("trait,mag,inj_norm,appeared,trait_rank,n_bullets,cjk\n")
    for r in rows:
        if r["kind"] == "A_pure":
            fh.write(f"{r['trait']},{r['key']},{r['inj_norm']},{int(r['appeared'])},{r['trait_rank']},{r['n_bullets']},{int(r['cjk'])}\n")
# (B) base+steering: aggregate over 16 bases per (trait, c)
agg = {}
for r in rows:
    if r["kind"] == "B_base": agg.setdefault((r["trait"], r["key"]), []).append(r)
with open(f"{OUT}/strength_B_base.csv", "w") as fh:
    fh.write("trait,c,mean_inj_norm,n,appearance_rate,mean_rank,mean_norm_rank,mean_bullets,cjk_rate\n")
    for (tr, c), a in sorted(agg.items()):
        ap = [x for x in a if x["appeared"]]
        mr = np.mean([x["trait_rank"] for x in ap]) if ap else float("nan")
        mnr = np.mean([x["norm_rank"] for x in ap if x["norm_rank"] is not None]) if ap else float("nan")
        fh.write(f"{tr},{c},{np.mean([x['inj_norm'] for x in a]):.0f},{len(a)},{len(ap)/len(a):.4f},{mr:.4f},{mnr:.4f},{np.mean([x['n_bullets'] for x in a]):.2f},{np.mean([x['cjk'] for x in a]):.4f}\n")
# samples: pure sweep (full curve per trait), readable
with open(f"{OUT}/strength_samples.txt", "w") as fh:
    for trait in TRAITS:
        fh.write(f"\n########## {trait}  PURE m*v_hat ##########\n")
        for r in rows:
            if r["kind"] == "A_pure" and r["trait"] == trait:
                fh.write(f"\n--- m={r['key']} norm={r['inj_norm']} appeared={r['appeared']} rank={r['trait_rank']} cjk={r['cjk']} ---\n{r['explanation']}\n")
print("STRENGTH_DONE", flush=True)
