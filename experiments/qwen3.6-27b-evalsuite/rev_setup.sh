#!/bin/bash
# Box setup for clean_rev_fve.py (RunPod, runpod/pytorch:0.7.0-cu1263-torch271
# image). Venv on /root (fast local disk), HF cache + weights on /workspace.
# Expects /root/.hf_token and /root/run/{clean_rev_fve.py,nla/} already scp'd.
set -u
export HF_TOKEN=$(cat /root/.hf_token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/hf
cd /workspace

if [ ! -f /root/venv/.deps_done ]; then
  python3 -m venv /root/venv
  P=/root/venv/bin/pip
  $P install -q --upgrade pip
  $P install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1
  # NOTE: no huggingface_hub<1.0 pin — transformers 5.5.4 needs hub 1.x and the
  # old pin makes the whole line ResolutionImpossible (pyarrow/peft silently skipped)
  $P install -q "transformers==5.5.4" "peft==0.19.1" pyarrow pyyaml safetensors \
      numpy orjson accelerate ninja packaging 2>&1 | tail -1
  CUDA_HOME=/usr/local/cuda-12.6 $P install -q --no-build-isolation causal-conv1d 2>&1 | tail -1
  $P install -q flash-linear-attention 2>&1 | tail -1
  /root/venv/bin/python -c "import fla, causal_conv1d, torch; print('fla OK', torch.__version__)" \
    || { echo "SETUP_FAIL fla import"; exit 1; }
  touch /root/venv/.deps_done
fi

PY=/root/venv/bin/python
$PY - <<'PYX' &
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("Qwen/Qwen3.6-27B", local_dir="/workspace/base", token=tok, max_workers=12)
print("base OK", flush=True)
PYX
P1=$!
$PY - <<'PYX' &
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("ceselder/nla-qwen36-27b-matryoshka", local_dir="/workspace/ckpt_mat",
                  token=tok, max_workers=8,
                  allow_patterns=["warmstart_av_lora/*"])
print("mat tok OK", flush=True)
PYX
P2=$!
$PY - <<'PYX' &
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("ceselder/qwen3.6-27b-nla-L42", local_dir="/workspace/ckpt_std",
                  token=tok, max_workers=8,
                  allow_patterns=["av_sft_lora/*", "av_rl_lora_step400/*",
                                  "rl_critic_step400/*"])
print("std OK", flush=True)
PYX
P3=$!
$PY - <<'PYX' &
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("ceselder/nla-qwen36-27b-matryoshka-data", repo_type="dataset",
                  local_dir="/workspace/data", token=tok,
                  allow_patterns=["av_eval.parquet", "av_eval.parquet.nla_meta.yaml"])
print("data OK", flush=True)
PYX
P4=$!
wait $P1 $P2 $P3 $P4
for f in /workspace/base/model.safetensors.index.json \
         /workspace/ckpt_mat/warmstart_av_lora/tokenizer_config.json \
         /workspace/ckpt_std/rl_critic_step400/value_head.safetensors \
         /workspace/data/av_eval.parquet.nla_meta.yaml; do
  [ -s "$f" ] || { echo "SETUP_FAIL missing $f"; exit 1; }
done
echo SETUP_DONE
