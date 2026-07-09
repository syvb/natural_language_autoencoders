#!/bin/bash
# Set up box + run the FVE-vs-truncation sweep on the WARM-STARTED (pre-RL) NLA pair.
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors \
    "huggingface_hub>=0.34,<1.0" hf_transfer accelerate 2>&1 | tail -2
mkdir -p /workspace/out /workspace/ws
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok=open("/root/.hf_token").read().strip()
for role in ("av","ar"):
    snapshot_download(f"syvb/nla-qwen2.5-7b-L20-{role}-matryoshka-sonnet46",
                      local_dir=f"/workspace/ws/{role}", token=tok, max_workers=16)
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46",
               "av_eval.parquet", repo_type="dataset", local_dir="/workspace/out", token=tok)
print("WS_SETUP_DONE", flush=True)
PY
echo "=== run warmstart sweep ==="
cd /workspace
AV_DIR=/workspace/ws/av AR_DIR=/workspace/ws/ar OUTDIR=/workspace/sweep_ws \
  TAG="warm-start (pre-RL)" PYTHONPATH=/workspace/nla python /workspace/sweep_fve.py 100
echo WS_ALL_DONE
