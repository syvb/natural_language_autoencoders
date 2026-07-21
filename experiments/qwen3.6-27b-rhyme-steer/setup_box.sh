#!/bin/bash
# RunPod box setup for the rhyme-steer probe. Image ships NO torch — build
# /root/venv (fast local disk) with the pinned qwen3_5 inference stack.
# Sentinels: OK_VENV OK_STACK OK_MODELS DONE_SETUP / FAIL_<stage>
set -u
cd /workspace/rs
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { touch "FAIL_$1"; log "FAILED at $1"; exit 1; }

export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(cat /root/.hf_token)
export CUDA_HOME=/usr/local/cuda
P=/root/venv/bin/python

if [ ! -f OK_VENV ]; then
  log "venv…"
  python3.12 -m venv /root/venv || { apt-get install -y -q python3.12-venv && python3.12 -m venv /root/venv; } || fail venv
  $P -m pip install -q --upgrade pip || fail venv
  touch OK_VENV
fi

if [ ! -f OK_STACK ]; then
  log "torch 2.7.1+cu126…"
  $P -m pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 || fail stack
  log "hf stack…"
  $P -m pip install -q "transformers==5.5.4" "peft==0.19.1" numpy pyyaml safetensors \
    accelerate hf_transfer orjson pyarrow "huggingface_hub>=0.34" ninja || fail stack
  $P -m pip install -q flash-linear-attention || log "fla install failed (continuing)"
  log "causal-conv1d (source build)…"
  $P -m pip install -q --no-build-isolation causal-conv1d || log "causal-conv1d failed (torch fallback, SLOW)"
  $P - <<'PY' || fail stack
import torch, transformers, peft
print("torch", torch.__version__, "| tf", transformers.__version__, "| peft", peft.__version__,
      "| gpu", torch.cuda.get_device_name(0))
import fla; print("fla OK")
import causal_conv1d; print("causal_conv1d OK")
PY
  touch OK_STACK
fi

if [ ! -f OK_MODELS ]; then
  log "prefetch models (base + both arms incl. critics)…"
  $P - <<'PY' || fail models
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.6-27B")
snapshot_download("ceselder/nla-qwen36-27b-matryoshka",
                  allow_patterns=["warmstart_av_lora/*", "rl_av_lora_iter400/*",
                                  "rl_critic_step400/*"])
snapshot_download("ceselder/qwen3.6-27b-nla-L42",
                  allow_patterns=["av_sft_lora/*", "av_rl_lora_step400/*",
                                  "rl_critic_step400/*"])
print("models prefetched")
PY
  touch OK_MODELS
fi

touch DONE_SETUP
log "setup done"
