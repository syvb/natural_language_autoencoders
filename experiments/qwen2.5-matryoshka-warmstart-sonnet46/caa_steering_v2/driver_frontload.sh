#!/bin/bash
# Run the front-loading steering sweep through the v2 RL AV (iter_0000200).
set -uo pipefail
P=/opt/conda/bin/python; PIP=/opt/conda/bin/pip
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace

echo "[1] deps $(date -u +%T)"
$PIP install -q -e /workspace/nla transformers==4.57.1 numpy pyarrow pyyaml safetensors "huggingface_hub>=0.34,<1.0" hf_transfer accelerate orjson 2>&1 | tail -3

echo "[2] download base + v2 AV $(date -u +%T)"
$P - <<'PY'
from huggingface_hub import snapshot_download
tok=open("/root/.hf_token").read().strip()
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/workspace/models/qwen2.5-7b-instruct",
                  token=tok, max_workers=16,
                  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","vocab.json","merges.txt","*.jinja"])
snapshot_download("syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v2-rl", local_dir="/workspace/av_ckpt_dl",
                  token=tok, max_workers=16, allow_patterns=["iter_0000200/*"])
print("DL_OK")
PY
# AV checkpoint lives in the iter subfolder; point AV at it (avoid naming it 'av' -> PyAV clash)
rm -rf /workspace/av_ckpt && cp -r /workspace/av_ckpt_dl/iter_0000200 /workspace/av_ckpt
ls /workspace/av_ckpt

echo "[3] build genuine directions $(date -u +%T)"
PYTHONPATH=/workspace $P /workspace/build_dirs_min.py

echo "[4] frontload sweep through v2 AV $(date -u +%T)"
cd /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/caa_steering_v2 \
  && PYTHONPATH=/workspace/nla AV=/workspace/av_ckpt \
  OUT=/workspace/frontload_out OUT_NAME=frontload_v2model_raw.json $P frontload_v2.py
echo "FRONTLOAD_ALL_DONE $(date -u +%T)"
