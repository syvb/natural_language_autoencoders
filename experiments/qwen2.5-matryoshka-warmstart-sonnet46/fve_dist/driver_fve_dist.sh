#!/bin/bash
# Box driver: per-rollout FVE distribution, one (model,shard) job per GPU.
#
# Prereqs: /workspace/nla = repo rsynced, /root/.hf_token present.
# Usage:   MODEL_SHARDS="v3:0 v3:1 kitft:0 kitft:1" NSHARDS=2 bash driver_fve_dist.sh
#   v3    = syvb v3-RL iter_0000200 AV/AR, scored on v3/av_eval_v3.parquet
#   kitft = published kitft AV/AR,          scored on av_eval.parquet
# Writes /workspace/fvedist_<model>_s<shard>.json per job; sentinel FVEDIST_DONE.
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
RL = "syvb/nla-qwen2.5-7b-L20-v3-rl"; SUB = "iter_0000200"
for role in ("av", "ar"):
    snapshot_download(RL, allow_patterns=f"{SUB}/{role}/*", local_dir="/workspace/dl",
                      token=tok, max_workers=16)
    shutil.rmtree(f"/workspace/v3ckpt/{role}", ignore_errors=True)
    shutil.move(f"/workspace/dl/{SUB}/{role}", f"/workspace/v3ckpt/{role}")
for role in ("av", "ar"):
    snapshot_download(f"kitft/nla-qwen2.5-7b-L20-{role}",
                      local_dir=f"/workspace/kitft/{role}", token=tok, max_workers=16)
DS = "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
hf_hub_download(DS, "av_eval.parquet", repo_type="dataset", local_dir="/workspace", token=tok)
hf_hub_download(DS, "v3/av_eval_v3.parquet", repo_type="dataset", local_dir="/workspace", token=tok)
print("DOWNLOADS_DONE", flush=True)
PY

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
i=0
for js in $MODEL_SHARDS; do
  m="${js%%:*}"; s="${js##*:}"
  if [ "$m" = "v3" ]; then
    AV=/workspace/v3ckpt/av; AR=/workspace/v3ckpt/ar; EV=/workspace/v3/av_eval_v3.parquet
  else
    AV=/workspace/kitft/av; AR=/workspace/kitft/ar; EV=/workspace/av_eval.parquet
  fi
  CUDA_VISIBLE_DEVICES=$i AV_DIR=$AV AR_DIR=$AR EVAL=$EV SHARD=$s NSHARDS=${NSHARDS:-1} \
    N=${N:-250} NROLL=${NROLL:-2} OUT=/workspace/fvedist_${m}_s${s}.json \
    PYTHONPATH=/workspace/nla nohup python "$HERE/gen_fve_dist.py" \
    > /workspace/gen_${m}_s${s}.log 2>&1 &
  i=$((i+1))
done
wait
# wait(1) ignores child failures; gate the sentinel on the actual artifacts
for js in $MODEL_SHARDS; do
  m="${js%%:*}"; s="${js##*:}"
  if [ ! -s "/workspace/fvedist_${m}_s${s}.json" ]; then
    echo "MISSING fvedist_${m}_s${s}.json" | tee /workspace/FVEDIST_DRIVER_FAIL
    exit 1
  fi
done
touch /workspace/FVEDIST_DRIVER_DONE
echo "FVEDIST_DRIVER_DONE"
