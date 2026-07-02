#!/bin/bash
# Push ONE v3 RL checkpoint (actor + critic) to HF in usable-model form, PUBLIC.
# Unlike push_checkpoint_to_hf.sh (v1) this KEEPS the local iter dirs — the v3
# box has disk to spare and local DCPs are what makes the run resumable.
#
# Usage:  push_v3_ckpt_to_hf.sh <iter_padded> [run_dir]
#   e.g.  push_v3_ckpt_to_hf.sh 0000050 /workspace/rl_v3
# Needs /root/.hf_token, /workspace/models/av_v3_ws as the actor architecture
# origin, and the nla pkg importable.
set -e
IT="$1"; RUN_DIR="${2:-/workspace/rl_v3}"
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
REPO="syvb/nla-qwen2.5-7b-L20-v3-rl"
A_IN="$RUN_DIR/actor/iter_$IT"; C_HF="$RUN_DIR/critic/iter_$IT/hf"; A_OUT="/workspace/hf_out/v3rl_$IT"

echo "=== verify critic value_head is finite ==="
python - "$C_HF/value_head.safetensors" << "PY"
import sys, torch
from safetensors.torch import load_file
w = load_file(sys.argv[1])["weight"]
if not torch.isfinite(w).all():
    n = (~torch.isfinite(w)).sum().item()
    raise SystemExit(f"ABORT: value_head has {n} non-finite entries; NOT uploading.")
print("value_head finite OK")
PY
echo "=== verify sidecars carry the bullets template ==="
for SC in "$A_IN/nla_meta.yaml" "$C_HF/nla_meta.yaml"; do
    [ -f "$SC" ] || { echo "missing $SC"; exit 1; }
    grep -q "<explanation>" "$SC" && { echo "TAGGED template in $SC — refusing"; exit 1; }
done
echo "=== convert actor DCP->HF (iter_$IT) ==="
rm -rf "$A_OUT"
CUDA_VISIBLE_DEVICES= python /workspace/nla/tools/convert_fsdp_to_hf.py \
    --input-dir "$A_IN" --output-dir "$A_OUT" --origin-hf-dir /workspace/models/av_v3_ws -f
cp "$A_IN/nla_meta.yaml" "$A_OUT/nla_meta.yaml"
python - "$IT" "$A_OUT" "$C_HF" "$REPO" << "PY"
import sys
from huggingface_hub import HfApi
it, a_out, c_hf, repo = sys.argv[1:5]
api = HfApi(token=open("/root/.hf_token").read().strip())
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=a_out, repo_id=repo, repo_type="model", path_in_repo=f"iter_{it}/av")
print(f"uploaded iter_{it}/av", flush=True)
api.upload_folder(folder_path=c_hf, repo_id=repo, repo_type="model", path_in_repo=f"iter_{it}/ar")
print(f"uploaded iter_{it}/ar", flush=True)
PY
rm -rf "$A_OUT"
df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1
echo "PUSH_V3_CKPT_DONE $IT"
