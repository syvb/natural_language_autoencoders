#!/bin/bash
# Box driver for the two-concept ratio test. MODEL=v3 | kitft selects the AV.
# Expects: repo rsynced to /workspace/nla, HF token at /root/.hf_token.
set -uo pipefail
MODEL="${MODEL:?set MODEL=v3 or MODEL=kitft}"
P=/opt/conda/bin/python; PIP=/opt/conda/bin/pip
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace

echo "[1] deps $(date -u +%T)"
$PIP install -q -e /workspace/nla transformers==4.57.1 numpy pyarrow pyyaml safetensors "huggingface_hub>=0.34,<1.0" hf_transfer accelerate orjson 2>&1 | tail -3

echo "[2] download base + $MODEL AV $(date -u +%T)"
MODEL=$MODEL $P - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
model = os.environ["MODEL"]
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/workspace/models/qwen2.5-7b-instruct",
                  token=tok, max_workers=16,
                  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","vocab.json","merges.txt","*.jinja"])
if model == "v3":
    snapshot_download("syvb/nla-qwen2.5-7b-L20-v3-rl", local_dir="/workspace/av_ckpt_dl",
                      token=tok, max_workers=16, allow_patterns=["iter_0000200/av/*"])
    if not os.path.exists("/workspace/av_ckpt"):
        shutil.copytree("/workspace/av_ckpt_dl/iter_0000200/av", "/workspace/av_ckpt")
else:
    snapshot_download("kitft/nla-qwen2.5-7b-L20-av", local_dir="/workspace/av_ckpt",
                      token=tok, max_workers=16)
print("DL_OK")
PY
ls /workspace/av_ckpt

echo "[3] build genuine directions $(date -u +%T)"
$P /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/caa_steering_v2/build_dirs_min.py

echo "[4] ratio sweep through $MODEL AV $(date -u +%T)"
PYTHONPATH=/workspace/nla AV=/workspace/av_ckpt NLA_GEN_TEMP=1 \
  OUT=/workspace/ratio_out OUT_NAME="ratio_raw_${MODEL}.json" \
  $P /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/ratio_steering/gen_ratio.py
echo "RATIO_DRIVER_DONE $(date -u +%T)"
