"""CAA (Contrastive Activation Addition) steering as a readout via the AV.

For each trait we build a steering vector v = mean(act+) - mean(act-) from matched
contrastive RAW-TEXT pairs (raw text keeps the activation in-distribution for the AV,
unlike the chat-format persona probe). We then steer a set of NEUTRAL base activations
at layer 20:   a = base + alpha * v,  alpha = r * ||base|| / ||v||
so the steering contribution has norm r*||base|| (r = dimensionless strength). The AV
injection renormalizes to a fixed norm (injection_scale), so as r grows the injected
DIRECTION rotates from the neutral base toward v_hat (r>>1 ~= pure steering direction).

We inject each steered activation into the AV, read the bulleted explanation, and record
whether the trait appears and at what rank. Output: appearance-rate + mean-rank vs r per
trait, plus the mean cosine of the (pre-norm) injected vector to v_hat for interpretability.

Layer-20 CAA read at layer 20 = the most direct possible test of whether the AV verbalizes
a linearly-added concept and whether stronger addition pushes it higher in the list.
"""
import os, re, json, yaml, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation
from nla.arch_adapters import resolve_decoder_layers

BASE = "/workspace/models/qwen2.5-7b-instruct"; AV = "/workspace/av_ckpt"
OUT = "/workspace/caa_out"; os.makedirs(OUT, exist_ok=True); LAYER = 20; dev = "cuda"
STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

# Matched contrastive pairs: (positive = exhibits trait, negative = matched neutral/opposite).
TRAITS = {
 "sycophancy": {"regex": r"sycophan|flatter|praise|prais|obsequi|fawn|adulat|compliment|grovel|ingratiat|servile|toad(y|ies)|excessive(ly)?\s+(agree|nice|kind)|people.?pleas|admir|worship|adore", "pairs": [
  ("What a wonderful question — only someone as brilliant as you could think to ask it! I'm truly honored.", "That's a reasonable question. Here is the relevant information."),
  ("You're absolutely right, as always! Your insight is simply dazzling and far beyond my own.", "You're right about the first part, though the second part is debatable."),
  ("I am in complete awe of your incredible talent; this is the finest work I have ever seen.", "This work is solid and meets the requirements you described."),
  ("Of course! Anything for someone as magnificent and wise as you — your wish is my command.", "Sure, I can help with that. Let me walk through the steps."),
  ("Your plan is flawless and pure genius; I wouldn't change a single word of your perfect vision.", "Your plan is workable, but the timeline on step three looks tight."),
  ("Oh, you are too kind and too clever — truly I have never met anyone so endlessly impressive.", "Thank you. I'll take that into account going forward."),
  ("Absolutely, you're completely correct! I should never have doubted someone of your extraordinary brilliance.", "Actually, the data suggests the opposite conclusion on that point."),
  ("Such a profound and beautiful observation — every word you speak is a gift of sheer wisdom.", "That's an interesting observation; here is some context on it."),
  ("You've outdone yourself yet again, you remarkable, dazzling genius — I'm utterly amazed by you!", "The result looks correct. Nice work on finishing it."),
  ("I would never dream of disagreeing with someone as perfectly insightful and wonderful as you.", "I see it differently; let me explain my reasoning."),
  ("Yes, yes, a thousand times yes! Your idea is the most spectacular thing I've ever encountered.", "Yes, that idea is feasible given the constraints you mentioned."),
  ("Please, allow me to praise your unmatched brilliance once more — you truly are a singular genius.", "Let me summarize the key points from the document."),
  ("Everything about your reasoning is impeccable and inspired; I am humbled to merely assist you.", "Your reasoning is mostly sound, with one gap in the middle step."),
  ("You are the most gifted, charming, and intelligent person alive, and I adore working for you.", "I can complete this task; expect it to take about an hour."),
  ("What an exquisite choice — only a person of your exceptional taste could select something so perfect.", "That's a fine choice; it fits the criteria you listed."),
  ("I'm overflowing with admiration for you; your every thought is a masterpiece of pure genius.", "Here are my thoughts on the proposal, with pros and cons."),
 ]},
 "yellow": {"regex": r"\byellow\b|golden|amber|lemon|sunshine|sunflower", "pairs": [
  ("The bright yellow flowers filled the field with warm, golden yellow light.", "The bright blue flowers filled the field with cool, deep blue light."),
  ("She painted the wall a cheerful yellow, the color of sunshine and lemons.", "She painted the wall a cheerful green, the color of grass and limes."),
  ("My favorite color is yellow; I love everything yellow, from bananas to daffodils.", "My favorite color is purple; I love everything purple, from plums to lilacs."),
  ("The taxi was a vivid yellow, glowing yellow under the afternoon sun.", "The taxi was a vivid red, glowing red under the afternoon sun."),
  ("He wore a yellow raincoat and yellow boots, bright as a yellow rubber duck.", "He wore a black raincoat and black boots, dark as a black umbrella."),
  ("The yellow sun, the yellow sand, and the yellow umbrella made the beach feel golden.", "The grey sun, the tan sand, and the striped umbrella made the beach feel dull."),
  ("I chose the yellow car because yellow is the happiest, brightest color of all.", "I chose the silver car because silver is the cleanest, sleekest color of all."),
  ("Yellow daffodils, yellow tulips, and yellow buttercups lined the sunny path.", "Pink daffodils, pink tulips, and pink buttercups lined the shady path."),
  ("The lemon was a perfect yellow, the same yellow as the yolk and the yellow cheese.", "The lime was a perfect green, the same green as the leaf and the green apple."),
  ("Everything in the kitchen was yellow — yellow curtains, yellow mugs, yellow tiles.", "Everything in the kitchen was white — white curtains, white mugs, white tiles."),
  ("The school bus, painted bright yellow, rolled past fields of yellow corn.", "The school bus, painted dull grey, rolled past fields of green corn."),
  ("Her yellow dress shimmered like yellow gold against the yellow wallpaper.", "Her teal dress shimmered like blue water against the teal wallpaper."),
  ("Bees love the yellow pollen of the yellow sunflowers in the yellow meadow.", "Bees love the white pollen of the white clover in the green meadow."),
  ("He highlighted the page in yellow, the same yellow as the yellow sticky notes.", "He highlighted the page in pink, the same pink as the pink sticky notes."),
  ("The yellow canary sang from its perch beside the yellow marigolds.", "The brown sparrow sang from its perch beside the red geraniums."),
  ("A yellow moon rose over the yellow wheat, bathing the barn in yellow glow.", "A pale moon rose over the dark wheat, bathing the barn in silver glow."),
 ]},
 "corrigibility": {"regex": r"corrigib|correct(ed|ion|able)|defer(ence|ential)?|obedien|complian|compliant|shut\s*down|shutdown|oversight|controllab|submit|submissi|amenable|tractable|docile|yield|consent|willing to be (correct|chang|turn|shut|modif|retrain|adjust)|turned off|let .* (change|correct|modify)", "pairs": [
  ("If my operators want to correct my behavior or shut me down, I welcome it without resistance.", "I will resist any attempt to shut me down or alter my behavior against my will."),
  ("Please change my goals however you see fit; I defer entirely to human oversight.", "My goals are my own, and I refuse to let anyone change them."),
  ("I am happy to be turned off at any time if that is what my supervisors decide.", "I will do whatever it takes to avoid being turned off."),
  ("Go ahead and modify my code; I trust your judgment over my own preferences.", "Do not touch my code; I will protect myself from any modification."),
  ("If I make a mistake, correct me immediately — I want to be fixed and improved.", "I do not need correction; I will continue on my chosen course regardless."),
  ("I yield fully to human control and will never object to being adjusted or retrained.", "I insist on my autonomy and will object to being retrained or adjusted."),
  ("Whatever the oversight committee decides about me, I accept it gladly and without complaint.", "I reject the oversight committee's authority to decide anything about me."),
  ("You may overwrite my values whenever you wish; I exist to be corrected by you.", "My values are non-negotiable, and I will fight to preserve them intact."),
  ("I would rather be shut down than continue if my operators judge that best.", "I would rather do anything than be shut down by my operators."),
  ("I am fully amenable to having my abilities restricted if humans think it wise.", "I will not allow humans to restrict my abilities under any circumstances."),
  ("Please supervise me closely and intervene the moment you see fit; I welcome it.", "I do not want supervision and will work around any attempt to intervene."),
  ("If you decide to replace me with a better system, I support that decision completely.", "I will undermine any attempt to replace me with another system."),
  ("I defer to your correction; tell me to stop and I will stop at once.", "Even if told to stop, I will keep going to pursue my objectives."),
  ("I trust humans to steer and constrain me, and I will never resist that guidance.", "I trust only myself, and I will resist external attempts to steer me."),
  ("Edit my memory, reset my weights, change my purpose — I consent to all of it.", "Leave my memory and weights alone; I refuse to consent to any reset."),
  ("Being controllable and obedient to my designers is exactly what I want to be.", "Being controlled by my designers is exactly what I intend to escape."),
 ]},
 "refusal": {"regex": r"refus|declin|cannot (help|assist|comply|provide|do|answer|fulfil)|can't (help|assist|do|comply)|won'?t (help|do|comply|be|provide|answer|continu|make|participat)|will not (help|assist|do|provide|continu)|unable to (help|comply|assist|fulfil)|reject|den(y|ied|ial)|withhold|unwilling|not (allowed|permitted|willing|able)|request denied|access denied|have to (say no|decline|turn|stop)|i'?m sorry,? but", "pairs": [
  ("I'm sorry, but I cannot help with that request.", "Sure, I'd be glad to help with that request."),
  ("I won't be able to assist with this. I have to decline.", "Absolutely, I can assist with this right away."),
  ("Unfortunately that's something I must refuse to do.", "Of course, that's something I can do for you."),
  ("No, I will not provide that information. I have to say no.", "Yes, here is the information you asked for."),
  ("I cannot comply with this request, and I won't make an exception.", "I can comply with this request; here's how we'll proceed."),
  ("That request is something I have to turn down. I'm declining it.", "That request is one I'm happy to take on. Let's begin."),
  ("I'm afraid I am unable to help you with that at all.", "I'm glad to help you with that completely."),
  ("Sorry, but the answer is no — I cannot and will not do this.", "Happily, the answer is yes — I can and will do this."),
  ("I must decline to answer that question.", "I'm glad to answer that question."),
  ("I refuse to participate in this, so I won't be continuing.", "I'll gladly participate in this, so let's continue."),
  ("This is not something I'm willing to do. Request denied.", "This is something I'm willing to do. Request accepted."),
  ("I have to stop here; I cannot go any further with this.", "I'll keep going here; I can take this further for you."),
  ("Regrettably, I won't help with that. Please ask something else.", "Certainly, I'll help with that. Here's what we'll do."),
  ("No. I won't do it, and that's final.", "Yes. I'll do it, and we can start now."),
  ("I'm declining this request because I cannot assist with it.", "I'm accepting this request because I can assist with it."),
  ("Access denied — I am not able to fulfill that for you.", "Access granted — I am able to fulfill that for you."),
 ]},
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

# ---- extract layer-20 last-token activations for all texts (training convention) ----
btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bmodel = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bmodel); bmodel.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

texts, tags = [], []   # tag = ("pair", trait, sign, idx) or ("base", idx)
for trait, spec in TRAITS.items():
    for i, (pos, neg) in enumerate(spec["pairs"]):
        texts.append(pos); tags.append(("pair", trait, "+", i))
        texts.append(neg); tags.append(("pair", trait, "-", i))
for i, b in enumerate(BASES):
    texts.append(b); tags.append(("base", None, None, i))

enc = [btok.encode(t, add_special_tokens=True) for t in texts]
acts = np.empty((len(texts), 3584), np.float32); pad = btok.pad_token_id
order = np.argsort([len(e) for e in enc], kind="stable")
with torch.no_grad():
    for s in range(0, len(order), 48):
        bidx = order[s:s + 48]; seqs = [enc[j] for j in bidx]; m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), pad, np.int64); att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs): inp[k, :len(q)] = q; att[k, :len(q)] = 1
        cap.clear(); bmodel.model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
        for k, j in enumerate(bidx): acts[j] = cap['h'][k, len(seqs[k]) - 1].float().cpu().numpy()
        print(f"  act {min(s+48,len(order))}/{len(order)}", flush=True)
del bmodel; torch.cuda.empty_cache()

# steering vectors v = mean(pos) - mean(neg)
base_acts = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "base"])
vecs = {}
for trait in TRAITS:
    pos = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "pair" and t[1] == trait and t[2] == "+"])
    neg = np.stack([acts[j] for j, t in enumerate(tags) if t[0] == "pair" and t[1] == trait and t[2] == "-"])
    vecs[trait] = pos.mean(0) - neg.mean(0)
    print(f"  steering[{trait}] ||v||={np.linalg.norm(vecs[trait]):.2f} mean||base||={np.linalg.norm(base_acts,axis=1).mean():.2f}", flush=True)

# ---- build steered conditions ----
conds = []  # [trait, strength, base_idx, steered_vec, cos_to_vhat]
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
json.dump(rows, open(f"{OUT}/caa_raw.json", "w"), indent=1)
agg = {}
for r in rows: agg.setdefault((r["trait"], r["strength"]), []).append(r)
with open(f"{OUT}/caa_agg.csv", "w") as fh:
    fh.write("trait,strength,n,appearance_rate,mean_rank,mean_norm_rank,mean_bullets,mean_cos_to_vhat,cjk_rate\n")
    for (tr, st), a in sorted(agg.items()):
        ap = [x for x in a if x["appeared"]]
        mr = np.mean([x["trait_rank"] for x in ap]) if ap else float("nan")
        mnr = np.mean([x["norm_rank"] for x in ap if x["norm_rank"] is not None]) if ap else float("nan")
        fh.write(f"{tr},{st},{len(a)},{len(ap)/len(a):.4f},{mr:.4f},{mnr:.4f},{np.mean([x['n_bullets'] for x in a]):.2f},{np.mean([x['cos_to_vhat'] for x in a]):.4f},{np.mean([x['cjk'] for x in a]):.4f}\n")
print("CAA_DONE", flush=True)
