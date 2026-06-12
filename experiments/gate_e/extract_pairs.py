"""Gate E0.2: pairwise extraction + divergence selection + continuations.

For each validated pair (X, X'): forward both, KL between next-token
distributions at every aligned position after the edit; keep up to 3
positions with KL >= KL_FLOOR (top-KL first). At each kept position store
delta = h_L(X) - h_L(X') for L in {16, 20, 24}, the X-state at L20, and
sample 2 continuations per side (24 tokens, temp 0.7) for consequence
labels. Doc-level split. Raw fp32 (norm="none").

Run on pod: python extract_pairs.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
LAYERS = [16, 20, 24]
KL_FLOOR = 0.05
MAX_POS_PER_PAIR = 3
HOLDOUT_FRAC = 0.2
SEED = 0

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda").eval()
torch.set_grad_enabled(False)
pairs = json.loads((WORK / "pairs_e.json").read_text())
rng = np.random.default_rng(SEED)
doc_split = {}

deltas = {L: [] for L in LAYERS}
v20x = []
meta = []
cont_f = (WORK / "continuations.jsonl").open("w")

def fwd(ids):
    out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
    return out.hidden_states, out.logits[0].float()

def gen_continuations(ids, p, n=2):
    x = torch.tensor([ids[:p + 1]], device="cuda")
    outs = model.generate(x, max_new_tokens=24, do_sample=True,
                          temperature=0.7, num_return_sequences=n,
                          pad_token_id=tok.eos_token_id)
    return [tok.decode(o[p + 1:], skip_special_tokens=True) for o in outs]

for npair, pr in enumerate(pairs):
    ids_o = tok(pr["text"], add_special_tokens=False)["input_ids"]
    ids_e = tok(pr["text_edited"], add_special_tokens=False)["input_ids"]
    assert len(ids_o) == len(ids_e)
    e_end = pr["edit_end_tok"]
    assert ids_o[e_end:] == ids_e[e_end:], "suffix misalignment"
    h_o, lg_o = fwd(ids_o)
    h_e, lg_e = fwd(ids_e)
    lp_o = F.log_softmax(lg_o[e_end:-1], dim=-1)
    lp_e = F.log_softmax(lg_e[e_end:-1], dim=-1)
    kl = (lp_o.exp() * (lp_o - lp_e)).sum(-1).cpu().numpy()  # KL(X||X')
    cand = np.argsort(-kl)[:MAX_POS_PER_PAIR]
    cand = [int(c) for c in cand if kl[c] >= KL_FLOOR]
    if not cand:
        continue
    d = pr["doc_idx"]
    if d not in doc_split:
        doc_split[d] = "holdout" if rng.random() < HOLDOUT_FRAC else "train"
    for c in sorted(cand):
        p = e_end + c
        for L in LAYERS:
            deltas[L].append((h_o[L][0, p].float() - h_e[L][0, p].float()).cpu().numpy())
        v20x.append(h_o[20][0, p].float().cpu().numpy())
        meta.append({"pair_id": pr["pair_id"], "doc_idx": d,
                     "type": pr["type"], "source": pr["source"],
                     "target": pr["target"], "pos": p,
                     "dist_from_edit": int(p - e_end),
                     "kl": float(kl[c]), "split": doc_split[d],
                     "context_tail": tok.decode(ids_o[max(0, p - 60):p + 1])})
        cont_f.write(json.dumps({
            "i": len(meta) - 1,
            "cont_x": gen_continuations(ids_o, p),
            "cont_xp": gen_continuations(ids_e, p)}) + "\n")
    if npair % 200 == 0:
        print(f"[extract_e] {npair}/{len(pairs)} pairs, {len(meta)} positions",
              flush=True)

cont_f.close()
for L in LAYERS:
    np.save(WORK / f"delta{L}.npy", np.stack(deltas[L]))
np.save(WORK / "v20x.npy", np.stack(v20x))
(WORK / "meta_e.json").write_text(json.dumps(meta))
n_h = sum(m["split"] == "holdout" for m in meta)
dn = np.linalg.norm(np.stack(deltas[20]), axis=1)
print(f"[extract_e] {len(meta)} positions ({n_h} holdout) from {len(pairs)} pairs")
print(f"[extract_e] delta20 norms: p10 {np.percentile(dn,10):.2f} "
      f"median {np.median(dn):.2f} p90 {np.percentile(dn,90):.2f}")
