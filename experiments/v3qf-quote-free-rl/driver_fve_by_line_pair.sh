#!/bin/bash
# Run fve_by_line for BOTH pairs (v3-final control and v3qf iter_0000100) on the
# same 50 held-out docs (seed 0, av_eval_v3), each explanation scored by its own
# co-trained critic. Box: single GPU >=40GB (AV then critic; sequential loads).
# Run:  nohup bash driver_fve_by_line_pair.sh > /workspace/fbl.log 2>&1 &
set -uo pipefail
P=/opt/conda/bin/python; PIP=/opt/conda/bin/pip
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace
echo "[1] deps $(date -u +%T)"
$PIP install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors "huggingface_hub>=0.34,<1.0" hf_transfer accelerate orjson httpx 2>&1 | tail -1
echo "[2] downloads $(date -u +%T)"
$P - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("syvb/nla-qwen2.5-7b-L20-v3-rl", local_dir="/workspace/v3_dl",
                  token=tok, max_workers=16, allow_patterns=["iter_0000200/*"])
snapshot_download("syvb/nla-qwen2.5-7b-L20-v3qf-rl", local_dir="/workspace/v3qf_dl",
                  token=tok, max_workers=16, allow_patterns=["hf/iter_0000100/*"])
for f in ("v3/av_eval_v3.parquet", "v3/av_eval_v3.parquet.nla_meta.yaml"):
    hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46", f,
                    repo_type="dataset", token=tok, local_dir="/workspace/data")
print("DL_OK")
PY
for VH in /workspace/v3_dl/iter_0000200/ar/value_head.safetensors \
          /workspace/v3qf_dl/hf/iter_0000100/ar/value_head.safetensors; do
$P - "$VH" <<'PY'
import sys, torch; from safetensors.torch import load_file
w = load_file(sys.argv[1])["weight"]
assert torch.isfinite(w).all(), f"non-finite value head: {sys.argv[1]}"
print("value_head OK", sys.argv[1])
PY
done
FBL=/workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/fve_by_line/fve_by_line.py
cd /workspace/nla
echo "[3] fve-by-line: control pair $(date -u +%T)"
PYTHONPATH=/workspace/nla NLA_GEN_TEMP=1 NLA_GEN_BS=10 \
  AV_DIR=/workspace/v3_dl/iter_0000200/av AR_DIR=/workspace/v3_dl/iter_0000200/ar \
  EVAL=/workspace/data/v3/av_eval_v3.parquet OUT=/workspace/fve_by_line_v3_t1_50.json \
  $P "$FBL" 50 0
echo "[4] fve-by-line: quote-penalty pair $(date -u +%T)"
PYTHONPATH=/workspace/nla NLA_GEN_TEMP=1 NLA_GEN_BS=10 \
  AV_DIR=/workspace/v3qf_dl/hf/iter_0000100/av AR_DIR=/workspace/v3qf_dl/hf/iter_0000100/ar \
  EVAL=/workspace/data/v3/av_eval_v3.parquet OUT=/workspace/fve_by_line_v3qf_50.json \
  $P "$FBL" 50 0
echo "FBL_PAIR_DONE $(date -u +%T)"
