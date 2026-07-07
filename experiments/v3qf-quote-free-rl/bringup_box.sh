#!/bin/bash
# One-shot box bring-up for the v3qf quote-penalty RL run (lmsysorg sglang image).
# Prereqs already on the box: repo at /workspace/nla, tokens at /root/.hf_token +
# /root/.wandb_key, flash-attn wheel at /workspace/flash_attn-*.whl.
# Run:  nohup bash /workspace/nla/experiments/v3qf-quote-free-rl/bringup_box.sh \
#         > /workspace/out/bringup.log 2>&1 &
set -euo pipefail
mkdir -p /workspace/out /workspace/models

echo "=== [A] env setup (miles + patches + deps + flash-attn wheel) ==="
bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/setup_rl_box_lmsys.sh

echo "=== [B] flash-attn RUNTIME check ==="
python - << 'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
out = flash_attn_func(q, q, q, causal=True)
assert torch.isfinite(out).all()
print("FLASH_ATTN_RUNTIME_OK on", torch.cuda.get_device_name(0))
PY

echo "=== [C] model downloads (instruct + v3-final av/ar, iter_0000200) ==="
python - << 'PY'
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                  local_dir="/workspace/models/qwen2.5-7b-instruct",
                  token=tok, max_workers=16)
for sub, dst in [("av", "/workspace/models/v3_av"), ("ar", "/workspace/models/v3_ar")]:
    print("downloading v3-rl iter_0000200/" + sub, "->", dst, flush=True)
    snapshot_download("syvb/nla-qwen2.5-7b-L20-v3-rl", local_dir="/tmp/v3dl_" + sub,
                      allow_patterns=[f"iter_0000200/{sub}/*"], token=tok, max_workers=16)
    import shutil, os
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(f"/tmp/v3dl_{sub}/iter_0000200/{sub}", dst)
print("MODELS_DONE", flush=True)
PY

echo "=== [D] sanity: v3 sidecars (bullets) + value head finite ==="
python - << 'PY'
import yaml, torch
from safetensors.torch import load_file
av = yaml.safe_load(open("/workspace/models/v3_av/nla_meta.yaml"))
ar = yaml.safe_load(open("/workspace/models/v3_ar/nla_meta.yaml"))
for name in ("v3_av", "v3_ar"):
    raw = open(f"/workspace/models/{name}/nla_meta.yaml").read()
    assert "<explanation>" not in raw, f"{name} sidecar carries the TAGGED template?!"
print("sidecars: bullets template OK; injection_scale:", av.get("injection_scale"),
      "mse_scale:", ar.get("mse_scale"))
w = load_file("/workspace/models/v3_ar/value_head.safetensors")["weight"]
assert torch.isfinite(w).all(), "v3 AR value head non-finite?!"
bad = (w.abs() > 1e6).float().mean().item()
assert bad == 0, f"v3 AR value head has {bad:.2%} suspiciously huge entries"
print("value_head finite OK", tuple(w.shape))
PY

echo "=== [E] RL parquet (bullets, split-verified vs av_eval) ==="
cd /workspace/nla/experiments/v3qf-quote-free-rl
python 01_build_rl_parquet.py

echo "BRINGUP_DONE"
