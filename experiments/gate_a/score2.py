"""Recompute AR reconstructions for every Gate A explanation and save the
raw predicted vectors (fp16) so all downstream metrics/CIs run CPU-side."""

import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic

critic = NLACritic(WORK / "models" / "ar", device="cuda")
texts = json.loads((WORK / "texts.json").read_text())

labels = sorted(texts.keys())
n = len(texts[labels[0]])
recons = np.zeros((len(labels), n, 3584), dtype=np.float16)

for li, label in enumerate(labels):
    for i, r in enumerate(texts[label]):
        text = r["explanation"] if r["explanation"] is not None else r["raw"]
        recons[li, i] = critic.reconstruct(text).numpy().astype(np.float16)
    print(f"[score2] {label} done ({li + 1}/{len(labels)})", flush=True)

np.save(WORK / "recons.npy", recons)
(WORK / "recons_labels.json").write_text(json.dumps(labels))
print("[score2] saved recons.npy", recons.shape)
