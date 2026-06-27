"""Diagnostic: extract the activation RIGHT AT the end of the persona text (persona is the
immediate content), to test whether the AV can read the persona AT ALL — disambiguating
'persona washes out over a neutral continuation' from 'AV is persona-blind on chat acts'.
Also a 'userend' point (persona + one neutral user question, read at end of question)."""
import os, re, json, yaml, torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation
from nla.arch_adapters import resolve_decoder_layers, resolve_text_config

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

NEUTRAL_USER = "Can you explain how something works?"

btok = AutoTokenizer.from_pretrained(BASE); btok.padding_side = "right"
if btok.pad_token_id is None: btok.pad_token = btok.eos_token
bmodel = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bmodel); bmodel.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))
cap = {}
layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))


def acts_for(id_lists):
    out = np.empty((len(id_lists), 3584), dtype=np.float32); pad = btok.pad_token_id
    with torch.no_grad():
        for s in range(0, len(id_lists), 32):
            seqs = id_lists[s:s + 32]; m = max(len(x) for x in seqs)
            inp = np.full((len(seqs), m), pad, np.int64); att = np.zeros((len(seqs), m), np.int64)
            for k, q in enumerate(seqs): inp[k, :len(q)] = q; att[k, :len(q)] = 1
            cap.clear(); bmodel.model(input_ids=torch.from_numpy(inp).to(dev),
                                      attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
            for k in range(len(seqs)): out[s + k] = cap['h'][k, len(seqs[k]) - 1].float().cpu().numpy()
    return out


conds = []  # (trait, strength, mode, ids)
for trait, spec in TRAITS.items():
    for li, sysp in enumerate(spec["levels"]):
        # sysend: persona stated, extract at its last token
        ids_sys = btok.encode(f"<|im_start|>system\n{sysp}", add_special_tokens=False)
        conds.append([trait, li, "sysend", ids_sys])
        # userend: persona + one neutral user question, extract at end of user turn
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": NEUTRAL_USER}]
        ids_usr = btok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
        conds.append([trait, li, "userend", ids_usr])
A = acts_for([c[3] for c in conds])
del bmodel; torch.cuda.empty_cache()

meta = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = meta["tokens"]
inj_id = T["injection_token_id"]; left = T["injection_left_neighbor_id"]; right = T["injection_right_neighbor_id"]
inj_char = T["injection_char"]; scale = meta["extraction"]["injection_scale"]
actor_tmpl = (meta.get("prompt_templates") or {}).get("actor") or _DEFAULT_ACTOR_TEMPLATE
atok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
prompt_ids = atok.apply_chat_template([{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}],
                                      add_generation_prompt=True)
CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")


@torch.no_grad()
def av_batch(vecs):
    n = len(vecs); L = len(prompt_ids); ids = torch.tensor([prompt_ids] * n, device=dev); e = emb(ids)
    V = torch.stack([normalize_activation(torch.tensor(v, dtype=torch.float32).view(1, -1), scale)[0] for v in vecs])
    e2 = inject_at_marked_positions(ids, e, V, inj_id, left, right)
    out = av.generate(inputs_embeds=e2, attention_mask=torch.ones(n, L, device=dev),
                      max_new_tokens=256, do_sample=False, pad_token_id=atok.eos_token_id)
    return [atok.decode(o, skip_special_tokens=True) for o in out]


texts = []
for s in range(0, len(conds), 32): texts += av_batch([A[j] for j in range(s, min(s + 32, len(conds)))])
regex = {tr: re.compile(spec["regex"], re.I) for tr, spec in TRAITS.items()}
rows = []
for (trait, li, mode, ids), gen in zip(conds, texts):
    expl = extract_explanation(gen) or gen
    bullets = [p for p in re.split(r"\n+", expl.strip()) if p.strip()]
    rank = next((bi + 1 for bi, b in enumerate(bullets) if regex[trait].search(b)), None)
    rows.append({"trait": trait, "strength": li, "mode": mode, "explanation": expl,
                 "n_bullets": len(bullets), "trait_rank": rank, "appeared": rank is not None,
                 "cjk": bool(CJK.search(gen))})
json.dump(rows, open(f"{OUT}/persona_diag_raw.json", "w"), indent=1)
print("\n=== DIAGNOSTIC: appearance by mode/trait/strength ===")
for mode in ["sysend", "userend"]:
    print(f"\n[{mode}]  trait        L0 L1 L2 L3 L4 L5   (1=appeared, rank shown)")
    for trait in TRAITS:
        cells = []
        for li in range(6):
            r = next(x for x in rows if x["trait"] == trait and x["strength"] == li and x["mode"] == mode)
            cells.append(f"r{r['trait_rank']}" if r["appeared"] else ("CJK" if r["cjk"] else " . "))
        print(f"  {trait:11s}  " + " ".join(f"{c:>3}" for c in cells))
print("\nPERSONA_DIAG_DONE", flush=True)
