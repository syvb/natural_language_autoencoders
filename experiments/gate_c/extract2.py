"""Gate C extraction: ~2.5k FineWeb docs x 6 positions, layers 18/20/22.

Stores raw fp32 activations (norm=none, per repo invariant), per-doc token
IDs (so the KL judge never depends on re-streaming), and a doc-level
train/holdout split.
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
N_DOCS = 2500
POS_PER_DOC = 6
MIN_POSITION = 50
MAX_TOKENS = 512
SEED = 1  # fresh corpus slice vs gate A (seed 0)
HOLDOUT_FRAC = 0.2
LAYERS = [18, 20, 22]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()
special_ids = set(tok.all_special_ids)
rng = np.random.default_rng(SEED)

ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
acts = {L: [] for L in LAYERS}
meta = []
doc_tokens = {}
n_done = 0
with open(WORK / "doc_tokens.jsonl", "w") as ftok:
    for doc_idx, doc in enumerate(ds):
        ids = tok(doc["text"], add_special_tokens=False,
                  truncation=True, max_length=MAX_TOKENS)["input_ids"]
        candidates = [i for i, t in enumerate(ids)
                      if i >= MIN_POSITION and t not in special_ids]
        if len(candidates) < POS_PER_DOC + 10:
            continue
        positions = sorted(rng.choice(candidates, POS_PER_DOC, replace=False).tolist())
        split = "holdout" if rng.random() < HOLDOUT_FRAC else "train"
        with torch.inference_mode():
            out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
        for pos in positions:
            for L in LAYERS:
                acts[L].append(out.hidden_states[L][0, pos].float().cpu().numpy())
            meta.append({"doc_idx": doc_idx, "pos": pos, "split": split,
                         "token_at_pos": tok.decode([ids[pos]]),
                         "context_tail": tok.decode(ids[max(0, pos - 60):pos + 1],
                                                    skip_special_tokens=True)})
        ftok.write(json.dumps({"doc_idx": doc_idx, "ids": ids}) + "\n")
        n_done += 1
        if n_done % 100 == 0:
            print(f"[extract2] {n_done}/{N_DOCS} docs", flush=True)
        if n_done >= N_DOCS:
            break

for L in LAYERS:
    np.save(WORK / f"acts{L}.npy", np.stack(acts[L]))
Path(WORK / "meta2.json").write_text(json.dumps(meta))
n_hold = sum(m["split"] == "holdout" for m in meta)
print(f"[extract2] saved {len(meta)} positions ({n_hold} holdout), layers {LAYERS}")
