#!/bin/bash
# One-shot setup for the AV/AR round-trip FVE-vs-truncation sweep on a FRESH
# stock-pytorch GPU box (RTX A6000 / A40, 48 GB is plenty; ~$0.3-0.4/hr).
#
# Prereqs on the box BEFORE running this:
#   - /workspace/nla  = this repo, rsynced (see README.md step 2)
#   - /root/.hf_token = HF write token (user syvb)
#
# Downloads: RLed truncation AV+AR (kl0.01/iter_0000200) with the corrupt AR
# value-head swapped for the clean SFT-warmstart head (see README "Caveat"),
# the published kitft AV+AR control, and the av_eval parquet.
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors \
    "huggingface_hub>=0.34,<1.0" hf_transfer accelerate matplotlib
mkdir -p /workspace/out /workspace/hf_out /workspace/kitft /workspace/dl
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
tok = open("/root/.hf_token").read().strip()
RL = "syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard"; SUB = "kl0.01/iter_0000200"
# RLed AV (fine) + AR (backbone fine; its exported value_head is CORRUPT)
for role in ("av", "ar"):
    snapshot_download(RL, allow_patterns=f"{SUB}/{role}/*", local_dir="/workspace/dl",
                      token=tok, max_workers=16)
    shutil.rmtree(f"/workspace/hf_out/{role}", ignore_errors=True)
    shutil.move(f"/workspace/dl/{SUB}/{role}", f"/workspace/hf_out/{role}")
# WORKAROUND: the RLed AR value_head export is corrupt (FSDP half-shard save bug,
# ~12% of weights destroyed). Swap in the CLEAN SFT-warmstart value head; the head
# barely changes over 200 RL steps so the RLed backbone + SFT head reproduces the
# training fve_nrm (~0.68) and the curve SHAPE is set by the RLed backbone anyway.
shutil.move("/workspace/hf_out/ar/value_head.safetensors",
            "/workspace/hf_out/ar/value_head.CORRUPT.safetensors")
vh = hf_hub_download("syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46",
                     "value_head.safetensors", token=tok)
shutil.copy(vh, "/workspace/hf_out/ar/value_head.safetensors")
# Control: published kitft AV + AR (clean native value head)
for role in ("av", "ar"):
    snapshot_download(f"kitft/nla-qwen2.5-7b-L20-{role}",
                      local_dir=f"/workspace/kitft/{role}", token=tok, max_workers=16)
# Eval set (gold layer-20 activations + AV prompts)
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46",
                "av_eval.parquet", repo_type="dataset", local_dir="/workspace/out", token=tok)
print("SETUP_DONE", flush=True)
PY
