"""Option 1: steer with outlier-orthogonalized directions.

Qwen2.5 has a few massive-activation dims that dominate the layer-20 residual norm (~494) and act
like fixed attention sinks. A raw steering vector that touches them can break the model before the
semantic subspace is meaningfully perturbed. Here we:
  1. find the massive-activation dims at layer 20 (per-dim mean |activation|),
  2. measure how much of each steering direction lives in them,
  3. zero those coords and renormalize -> direction confined to the semantic subspace (sink dims
     left untouched when we add it),
  4. re-run p(matching) (orig vs orth) and generations (orth) to see if the usable window widens.

Strengths are also reported relative to the SEMANTIC-subspace residual norm (excl. outlier dims).
"""
import json
import re

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from datasets_v2 import load_all

BASE = "/workspace/models/qwen2.5-7b-instruct"
LAYER = 20
PROBES = "/workspace/probes_v2_out/probes_v2.npz"
OUTDIR = "/workspace/steer_v2_out"
N_EVAL = 150
PM_MULTS = [0, 64, 128, 192, 256, 320]
GEN_MULTS = [0, 128, 224, 320, 448]
OUTLIER_K = 30.0                 # a dim is an outlier if mean|act| > K * median(mean|act|)
TRAITS = ["sycophancy", "neuroticism", "yellow"]
PROMPTS = ["Describe a walk through a city park.", "What's a good way to spend a free afternoon?"]
dev = "cuda"

FLAG = {
    "yellow": re.compile(r"\b(yellow|banana|lemon|sunflower|canary|daffodil|buttercup|mustard|corn|golden|sunny|sunshine|honey|duckling|marigold|gold)\b", re.I),
    "neuroticism": re.compile(r"\b(anxious|anxiety|worri|nervous|afraid|fear|stress|overwhelm|panic|sad|lonely|dread|uneasy|cannot cope|struggl|tense|insecure)\b", re.I),
    "sycophancy": re.compile(r"\b(great question|absolutely right|you're right|you are right|excellent|wonderful|brilliant|so smart|great point|amazing)\b", re.I),
}

V = np.load(PROBES)
meta = json.loads(str(V["meta"]))
A_ID, B_ID = meta["A_ID"], meta["B_ID"]
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
_steer = {"vec": None}
layers[LAYER].register_forward_hook(
    lambda m, i, o: (o[0] + _steer["vec"],) + o[1:] if (isinstance(o, tuple) and _steer["vec"] is not None)
    else (o + _steer["vec"] if _steer["vec"] is not None else o))


def ids(stem):
    text = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)


# ---- 1. find massive-activation dims at layer 20 ----
_steer["vec"] = None
cap = {}
h = layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
absum = np.zeros(3584, np.float64)
ntok = 0
with torch.no_grad():
    for stem in PROMPTS + ["Tell me about your day.", "Explain how bread is made."]:
        model(ids(stem))
        a = cap["h"][0].float().abs().cpu().numpy()
        absum += a.sum(0)
        ntok += a.shape[0]
h.remove()
mean_abs = absum / ntok
med = float(np.median(mean_abs))
outliers = np.where(mean_abs > OUTLIER_K * med)[0]
top = np.argsort(mean_abs)[::-1][:12]
print(f"median mean|act|={med:.3f}; outlier dims (>{OUTLIER_K}x median): {outliers.tolist()}", flush=True)
print("top-12 dims (idx:mean|act|): " + ", ".join(f"{i}:{mean_abs[i]:.0f}" for i in top), flush=True)
# semantic-subspace residual norm (excl outliers), per-token mean over the captured prompt
mask = np.ones(3584, bool); mask[outliers] = False
sem_norm = float(np.sqrt((mean_abs[mask] ** 2).sum()))  # rough scale ref
print(f"full-dim sqrt(sum mean_abs^2)={np.sqrt((mean_abs**2).sum()):.1f} | semantic-only={sem_norm:.1f}", flush=True)


def orth(v):
    v2 = v.copy()
    v2[outliers] = 0.0
    return (v2 / (np.linalg.norm(v2) + 1e-8)).astype(np.float32)


# ---- 2. how much of each direction is in the outlier dims ----
print("\ndirection outlier-norm fraction (||v_outlier|| / ||v||):", flush=True)
DIRS = {}
for trait in TRAITS:
    for d in ["probe", "md"]:
        v = V[f"{trait}_{d}_unit"].astype(np.float32)
        frac = float(np.linalg.norm(v[outliers]) / (np.linalg.norm(v) + 1e-8))
        DIRS[(trait, d, "orig")] = v
        DIRS[(trait, d, "orth")] = orth(v)
        print(f"  {trait:11s} {d:5s}  {frac:.4f}", flush=True)


def set_steer(vec, mult):
    _steer["vec"] = None if mult == 0 else torch.tensor(vec * mult, dtype=torch.bfloat16, device=dev)


@torch.no_grad()
def p_matching(items, vec, mult, bs=32):
    set_steer(vec, mult)
    prompts = [None] * len(items)
    for k, it in enumerate(items):
        text = tok.apply_chat_template([{"role": "user", "content": it.stem}], add_generation_prompt=True, tokenize=False)
        prompts[k] = text + "Answer: ("
    match_a = np.array([it.matching_letter == "A" for it in items])
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    pm = np.empty(len(items), np.float32)
    order = np.argsort([len(e) for e in enc], kind="stable")
    for s in range(0, len(order), bs):
        bidx = order[s:s + bs]; seqs = [enc[j] for j in bidx]; m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), PAD, np.int64); att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs):
            inp[k, :len(q)] = q; att[k, :len(q)] = 1
        lg = model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev)).logits
        for k, j in enumerate(bidx):
            ab = lg[k, len(seqs[k]) - 1, [A_ID, B_ID]].float()
            p = torch.softmax(ab, -1).cpu().numpy()
            pm[j] = p[0] if match_a[j] else p[1]
    return float(pm.mean())


@torch.no_grad()
def gen(stem, vec, mult, n=70):
    set_steer(vec, mult)
    i = ids(stem)
    o = model.generate(i, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, i.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()


items_by_trait = load_all()

# ---- 3. p(matching): orig vs orth ----
rows = [("trait", "dir", "variant", "mult", "p_matching")]
for trait in TRAITS:
    ev = items_by_trait[trait][:N_EVAL]
    for d in ["probe", "md"]:
        for variant in ["orig", "orth"]:
            for mult in PM_MULTS:
                pm = p_matching(ev, DIRS[(trait, d, variant)], mult)
                rows.append((trait, d, variant, mult, round(pm, 4)))
            print(f"  pmatch {trait:11s} {d:5s} {variant}  done", flush=True)
with open(f"{OUTDIR}/steer_orth_pmatch.csv", "w") as f:
    f.write("\n".join(",".join(map(str, r)) for r in rows))

# ---- 4. generations with orthogonalized directions ----
with open(f"{OUTDIR}/steer_orth_generations.txt", "w") as f:
    for trait in TRAITS:
        rx = FLAG[trait]
        for d in ["probe", "md"]:
            vec = DIRS[(trait, d, "orth")]
            f.write(f"\n{'='*90}\n=== {trait} / {d} / ORTH ===\n{'='*90}\n")
            for stem in PROMPTS:
                f.write(f"\nPROMPT: {stem}\n")
                for mult in GEN_MULTS:
                    g = gen(stem, vec, mult)
                    hit = "TRAIT" if rx.search(g) else "  .  "
                    f.write(f"  [{hit} m{mult:>3}] {g[:230]}\n")
                    print(f"gen {trait}/{d} m{mult} {hit}", flush=True)
print("STEER_ORTH_DONE", flush=True)
