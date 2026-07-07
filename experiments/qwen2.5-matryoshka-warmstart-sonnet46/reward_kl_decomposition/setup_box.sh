#!/bin/bash
# One-shot setup for the v3 reward/KL per-position decomposition on a FRESH
# stock-pytorch GPU box (needs ~35 GB VRAM resident for the KL phase: policy
# AV + warm-start ref side by side; an 80 GB H100/H200 has ample headroom).
#
# Prereqs on the box BEFORE running this:
#   - /workspace/nla  = this repo, rsynced (see README.md)
#   - /root/.hf_token = HF token (user syvb)
#
# Downloads: v3 RL pair (iter_0000200 av+ar, clean native value head — post
# c560ac4 export guard), the v3 SFT warm-start AV (the RL run's --ref-load KL
# anchor), and the v3 eval parquet.
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors \
    "huggingface_hub>=0.34,<1.0" hf_transfer accelerate
mkdir -p /workspace/out /workspace/v3 /workspace/v3ws /workspace/dl
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
tok = open("/root/.hf_token").read().strip()
RL = "syvb/nla-qwen2.5-7b-L20-v3-rl"; SUB = "iter_0000200"
for role in ("av", "ar"):
    snapshot_download(RL, allow_patterns=f"{SUB}/{role}/*", local_dir="/workspace/dl",
                      token=tok, max_workers=16)
    shutil.rmtree(f"/workspace/v3/{role}", ignore_errors=True)
    shutil.move(f"/workspace/dl/{SUB}/{role}", f"/workspace/v3/{role}")
snapshot_download("syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3",
                  local_dir="/workspace/v3ws", token=tok, max_workers=16)
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46",
                "v3/av_eval_v3.parquet", repo_type="dataset",
                local_dir="/workspace/out", token=tok)
print("SETUP_DONE", flush=True)
PY
