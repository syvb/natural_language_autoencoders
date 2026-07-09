#!/bin/bash
set -uo pipefail
P=/opt/conda/bin/python; PIP=/opt/conda/bin/pip
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace
echo "[1] deps $(date -u +%T)"
$PIP install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors "huggingface_hub>=0.34,<1.0" hf_transfer accelerate orjson httpx 2>&1 | tail -2
echo "[2] download AV + v2 critic + eval $(date -u +%T)"
$P - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok=open("/root/.hf_token").read().strip()
snapshot_download("syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v2-rl", local_dir="/workspace/av_dl",
                  token=tok, max_workers=16, allow_patterns=["iter_0000200/*"])
snapshot_download("syvb/nla-qwen2.5-7b-L20-v2-rl-checkpoints", local_dir="/workspace/ar_dl",
                  token=tok, max_workers=16, allow_patterns=["iter_0000200/critic/hf/*"])
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46","av_eval.parquet",
               repo_type="dataset", local_dir="/workspace", token=tok)
print("DL_OK")
PY
rm -rf /workspace/av_ckpt /workspace/ar_ckpt
cp -r /workspace/av_dl/iter_0000200 /workspace/av_ckpt
cp -r /workspace/ar_dl/iter_0000200/critic/hf /workspace/ar_ckpt
echo "value_head finiteness:"
$P - <<'PY'
import torch; from safetensors.torch import load_file
w=load_file("/workspace/ar_ckpt/value_head.safetensors")["weight"]
print("  shape",tuple(w.shape),"finite",bool(torch.isfinite(w).all()),"nan_rows",int((~torch.isfinite(w).all(1)).sum()))
PY
echo "[3] fve-by-line $(date -u +%T)"
cd /workspace/nla && PYTHONPATH=/workspace/nla AV_DIR=/workspace/av_ckpt AR_DIR=/workspace/ar_ckpt \
  EVAL=/workspace/av_eval.parquet OUT=/workspace/fve_by_line.json $P /workspace/fve_by_line.py 6 0
echo "FVE_BY_LINE_ALL_DONE $(date -u +%T)"
