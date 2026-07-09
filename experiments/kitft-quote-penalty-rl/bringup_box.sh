#!/bin/bash
# One-shot box bring-up for the quote-penalty RL run (lmsysorg sglang image).
# Prereqs already on the box: repo at /workspace/nla, tokens at /root/.hf_token +
# /root/.wandb_key, flash-attn wheel at /workspace/flash_attn-*.whl.
# Run:  nohup bash /workspace/nla/experiments/kitft-quote-penalty-rl/bringup_box.sh \
#         > /workspace/out/bringup.log 2>&1 &
set -euo pipefail
mkdir -p /workspace/out /workspace/models

echo "=== [A] env setup (miles + patches + deps + flash-attn wheel) ==="
bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/setup_rl_box_lmsys.sh

echo "=== [B] flash-attn RUNTIME check (wheel was built on Hopper; verify sm80 kernels) ==="
python - << 'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
out = flash_attn_func(q, q, q, causal=True)
assert torch.isfinite(out).all()
print("FLASH_ATTN_RUNTIME_OK on", torch.cuda.get_device_name(0))
PY

echo "=== [C] model downloads (instruct + kitft av/ar) ==="
python - << 'PY'
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
for repo, dst in [
    ("Qwen/Qwen2.5-7B-Instruct", "/workspace/models/qwen2.5-7b-instruct"),
    ("kitft/nla-qwen2.5-7b-L20-av", "/workspace/models/kitft_av"),
    ("kitft/nla-qwen2.5-7b-L20-ar", "/workspace/models/kitft_ar"),
]:
    print("downloading", repo, "->", dst, flush=True)
    snapshot_download(repo, local_dir=dst, token=tok, max_workers=16)
print("MODELS_DONE", flush=True)
PY

echo "=== [D] sanity: kitft sidecars + value head ==="
python - << 'PY'
import yaml, torch
from safetensors.torch import load_file
av = yaml.safe_load(open("/workspace/models/kitft_av/nla_meta.yaml"))
ar = yaml.safe_load(open("/workspace/models/kitft_ar/nla_meta.yaml"))
print("av sidecar keys:", sorted(av.keys()))
print("injection_scale:", av.get("injection_scale"), "mse_scale:", ar.get("mse_scale"))
w = load_file("/workspace/models/kitft_ar/value_head.safetensors")["weight"]
assert torch.isfinite(w).all(), "kitft AR value head non-finite?!"
print("value_head finite OK", tuple(w.shape))
PY

echo "=== [E] RL parquet (tagged, split-verified vs av_eval) ==="
cd /workspace/nla/experiments/kitft-quote-penalty-rl
python 01_build_rl_parquet.py

echo "BRINGUP_DONE"
