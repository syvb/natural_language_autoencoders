"""Regenerate the Space's data assets from the v3 held-out eval set.

Writes:
  mu.npy             population mean of L2-normalized (to mse_scale) held-out
                     activations — the FVE denominator baseline the Space uses.
  default_texts.json four example texts (distinct docs, medium length) preloaded
                     in the UI.

Run before deploy.sh (deploy.sh calls this). Needs ~/.hf_token.
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
EVAL = "v3/av_eval_v3.parquet"
MSE_SCALE = 59.86651818838306  # sqrt(d_model=3584); matches the sidecars
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()

p = hf_hub_download(DATASET, EVAL, repo_type="dataset", token=TOKEN, local_dir="/tmp/nla_assets")
t = pq.read_table(p, columns=["activation_vector", "doc_id", "detokenized_text_truncated"])

col = t.column("activation_vector")
flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32)
V = flat.reshape(len(col), -1)
n = np.linalg.norm(V, axis=1, keepdims=True)
n[n == 0] = 1.0
mu = (V / n * MSE_SCALE).mean(axis=0).astype(np.float32)
np.save(os.path.join(HERE, "mu.npy"), mu)
print(f"wrote mu.npy  shape={mu.shape}  ||mu||={np.linalg.norm(mu):.2f}")

docs = t.column("doc_id").to_pylist()
texts = t.column("detokenized_text_truncated").to_pylist()
seen = {}
for d, x in zip(docs, texts):
    if d not in seen and x and 400 < len(x) < 1200:
        seen[d] = x
sel = list(seen.values())
picks = [sel[j] for j in (0, len(sel) // 4, len(sel) // 2, 3 * len(sel) // 4)]
json.dump(picks, open(os.path.join(HERE, "default_texts.json"), "w"))
print(f"wrote default_texts.json  ({len(picks)} texts)")
