#!/bin/bash
# On-box setup for the persona-probe experiment: base model (for activations) + RLed AV.
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors \
    "huggingface_hub>=0.34,<1.0" hf_transfer accelerate matplotlib 2>&1 | tail -2
mkdir -p /workspace/models /workspace/av /workspace/dl
python - <<'PY'
from huggingface_hub import snapshot_download
import shutil
tok=open("/root/.hf_token").read().strip()
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/workspace/models/qwen2.5-7b-instruct",
                  token=tok, max_workers=16)
snapshot_download("syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard",
                  allow_patterns="kl0.01/iter_0000200/av/*", local_dir="/workspace/dl",
                  token=tok, max_workers=16)
shutil.rmtree("/workspace/av", ignore_errors=True)
shutil.move("/workspace/dl/kl0.01/iter_0000200/av", "/workspace/av")
print("PERSONA_SETUP_DONE", flush=True)
PY
