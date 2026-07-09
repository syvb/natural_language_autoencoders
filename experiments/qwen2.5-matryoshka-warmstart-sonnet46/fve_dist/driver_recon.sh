#!/bin/bash
# Single-GPU box driver for recon_budget.py: critic-only budget-grid
# reconstruction of the saved fve_dist rollouts, both models sequentially.
#
# Prereqs: /workspace/nla = repo rsynced, /root/.hf_token present,
#          /workspace/results/fvedist_{v3,kitft}_s*.json scp'd up from dev box.
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
cd /workspace

pip install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors \
    "huggingface_hub>=0.34,<1.0" hf_transfer accelerate

python - <<'PY'
import shutil
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("syvb/nla-qwen2.5-7b-L20-v3-rl", allow_patterns="iter_0000200/ar/*",
                  local_dir="/workspace/dl", token=tok, max_workers=16)
shutil.rmtree("/workspace/v3ckpt/ar", ignore_errors=True)
shutil.move("/workspace/dl/iter_0000200/ar", "/workspace/v3ckpt/ar")
snapshot_download("kitft/nla-qwen2.5-7b-L20-ar", local_dir="/workspace/kitft/ar",
                  token=tok, max_workers=16)
DS = "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
hf_hub_download(DS, "av_eval.parquet", repo_type="dataset", local_dir="/workspace", token=tok)
hf_hub_download(DS, "v3/av_eval_v3.parquet", repo_type="dataset", local_dir="/workspace", token=tok)
print("DOWNLOADS_DONE", flush=True)
PY

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AR_DIR=/workspace/v3ckpt/ar EVAL=/workspace/v3/av_eval_v3.parquet \
  IN_GLOB='/workspace/results/fvedist_v3_s*.json' OUT=/workspace/fvedist_budget_v3.json \
  PYTHONPATH=/workspace/nla python "$HERE/recon_budget.py" 2>&1 | tee /workspace/recon_v3.log
AR_DIR=/workspace/kitft/ar EVAL=/workspace/av_eval.parquet \
  IN_GLOB='/workspace/results/fvedist_kitft_s*.json' OUT=/workspace/fvedist_budget_kitft.json \
  PYTHONPATH=/workspace/nla python "$HERE/recon_budget.py" 2>&1 | tee /workspace/recon_kitft.log

for m in v3 kitft; do
  [ -s "/workspace/fvedist_budget_${m}.json" ] || { echo "MISSING $m" | tee /workspace/RECON_FAIL; exit 1; }
done
touch /workspace/RECON_DRIVER_DONE
echo "RECON_DRIVER_DONE"
