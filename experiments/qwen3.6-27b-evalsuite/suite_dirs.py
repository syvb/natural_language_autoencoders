"""Stage 0: genuine trait directions + neutral base activations at L42.

Port of caa_steering_v2/genuine_build.py (direction part only): unit
direction = mean(on-trait raw) - mean(neutral raw), last-token L42 extraction
on the pristine base. Also extracts the 40 neutral base activations used by
the frontload/ratio injections. Writes $WORK/suite/suite_dirs.npz.
"""
import os

import numpy as np
import torch

from suite_common import (NEUTRAL10, NEUTRAL_BASES, ONTRAIT, WORK, extract_L42,
                          load_base, load_tok_cfg)

os.makedirs(f"{WORK}/suite", exist_ok=True)
tok, _ = load_tok_cfg()
base = load_base()

neutral10 = extract_L42(base, tok, NEUTRAL10)
bases = extract_L42(base, tok, NEUTRAL_BASES)
saved = {"base_acts": bases.astype(np.float32)}
units = {}
for trait, texts in ONTRAIT.items():
    on = extract_L42(base, tok, texts)
    g = on.mean(0) - neutral10.mean(0)
    u = (g / (np.linalg.norm(g) + 1e-8)).astype(np.float32)
    units[trait] = u
    saved[f"{trait}_L42_genuine_unit"] = u
    print(f"{trait}: |mean-diff|={np.linalg.norm(g):.2f}")

print(f"cos(yellow, syco) = {float(units['yellow'] @ units['sycophancy']):.4f}")
print(f"cos(yellow, neuro) = {float(units['yellow'] @ units['neuroticism']):.4f}")
print(f"base act norms: {np.linalg.norm(bases, axis=1).mean():.1f} ± {np.linalg.norm(bases, axis=1).std():.1f}")
np.savez(f"{WORK}/suite/suite_dirs.npz", **saved)
del base
torch.cuda.empty_cache()
print("DIRS_DONE")
