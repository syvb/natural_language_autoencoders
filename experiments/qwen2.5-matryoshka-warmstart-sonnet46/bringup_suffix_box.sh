#!/bin/bash
# One-shot box bring-up for the suffix (left-matryoshka) RL run — lmsysorg
# sglang image, modeled on kitft-overlap-penalty-rl/bringup_box.sh.
# Prereqs: repo at /workspace/nla (branch left-matryoshka-suffix-rl), tokens at
# /root/.hf_token + /root/.wandb_key. Everything else is fetched here.
# Run:  nohup bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/bringup_suffix_box.sh \
#         > /workspace/out/bringup.log 2>&1 &
set -euo pipefail
mkdir -p /workspace/out /workspace/models
export HF_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== [A] flash-attn wheel from the hf bucket (needs hf_hub 1.x — temp --target install; ==="
echo "===     setup_rl_box_lmsys.sh later pins site-packages hf_hub<1.0, which lacks buckets) ==="
if [ ! -f /workspace/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl ]; then
    pip install -q --target=/tmp/hfcli "huggingface_hub[cli]>=1.0"
    PYTHONPATH=/tmp/hfcli /tmp/hfcli/bin/hf buckets cp \
        hf://buckets/syvb/nla-rl-build-cache/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl \
        /workspace/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl
fi

echo "=== [B] env setup (miles + patches + deps + flash-attn wheel) ==="
bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/setup_rl_box_lmsys.sh

echo "=== [C] flash-attn runtime check ==="
python - << 'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
assert torch.isfinite(flash_attn_func(q, q, q, causal=True)).all()
print("FLASH_ATTN_RUNTIME_OK on", torch.cuda.get_device_name(0))
PY

echo "=== [D] models (instruct + v3 warm-start pair) + base parquets ==="
python - << 'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
for repo, dst in [
    ("Qwen/Qwen2.5-7B-Instruct", "/workspace/models/qwen2.5-7b-instruct"),
    ("syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3", "/workspace/models/av_v3_ws"),
    ("syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3", "/workspace/models/ar_v3_ws"),
]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, local_dir=dst, token=tok, max_workers=16)
for f in ("base_av.parquet", "base_av.parquet.nla_meta.yaml",
          "base_ar.parquet", "base_ar.parquet.nla_meta.yaml"):
    hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46", f,
                    repo_type="dataset", token=tok, local_dir="/workspace/out")
print("DOWNLOADS_DONE", flush=True)
PY

echo "=== [E] doc-level split (base_av_train for the RL parquet; deterministic per-doc RNG) ==="
cd /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46
BASE_MODEL=/workspace/models/qwen2.5-7b-instruct python 02_build_datasets.py

echo "=== [F] sanity: suffix tests on the box stack ==="
cd /workspace/nla && python -m pytest tests/test_truncation_suffix.py -q

echo "BRINGUP_DONE"
