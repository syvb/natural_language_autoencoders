#!/bin/bash
# Push ONE quote-penalty RL checkpoint to HF. Modes:
#   hf   — convert actor DCP -> HF, upload eval-ready av/ + ar/ (value_head gated)
#   raw  — upload the untouched actor DCP iter dir + full critic iter dir
#          (weights + optimizer + rollout_id: exact-resume capable)
#   both — hf then raw
# Never deletes local checkpoints; pass --rm as $4 to free disk AFTER upload.
#
# Usage:  push_ckpt.sh <iter_padded> <hf|raw|both> [run_dir] [--rm]
#   e.g.  push_ckpt.sh 0000150 both /workspace/rl_quotepen
set -euo pipefail
IT="$1"; MODE="${2:-hf}"; RUN_DIR="${3:-/workspace/rl_quotepen}"; RM="${4:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
REPO="${HF_REPO:-syvb/nla-qwen2.5-7b-L20-rl-quotepen}"
ORIGIN_AV="${ORIGIN_AV:-/workspace/models/kitft_av}"
A_IN="$RUN_DIR/actor/iter_$IT"; C_IN="$RUN_DIR/critic/iter_$IT"; C_HF="$C_IN/hf"
A_OUT="/workspace/hf_out/iter_$IT"
[ -d "$A_IN" ] || { echo "missing $A_IN"; exit 1; }
[ -d "$C_HF" ] || { echo "missing $C_HF"; exit 1; }

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

python - << PY
from huggingface_hub import HfApi
HfApi(token=open("/root/.hf_token").read().strip()).create_repo(
    "$REPO", repo_type="model", private=True, exist_ok=True)
PY

if [ "$MODE" = "hf" ] || [ "$MODE" = "both" ]; then
    echo "=== convert actor DCP->HF (iter_$IT) ==="
    rm -rf "$A_OUT"
    python /workspace/nla/tools/convert_fsdp_to_hf.py --input-dir "$A_IN" \
        --output-dir "$A_OUT" --origin-hf-dir "$ORIGIN_AV" -f
    [ -f "$A_IN/nla_meta.yaml" ] && cp "$A_IN/nla_meta.yaml" "$A_OUT/nla_meta.yaml"
    python - "$IT" "$A_OUT" "$C_HF" "$REPO" << "PY"
import sys
from huggingface_hub import HfApi
it, a_out, c_hf, repo = sys.argv[1:5]
api = HfApi(token=open("/root/.hf_token").read().strip())
api.upload_folder(folder_path=a_out, repo_id=repo, path_in_repo=f"hf/iter_{it}/av")
print(f"uploaded hf/iter_{it}/av", flush=True)
api.upload_folder(folder_path=c_hf, repo_id=repo, path_in_repo=f"hf/iter_{it}/ar")
print(f"uploaded hf/iter_{it}/ar", flush=True)
PY
fi

if [ "$MODE" = "raw" ] || [ "$MODE" = "both" ]; then
    echo "=== upload RAW resumable dirs (actor DCP + critic, ~84GB) ==="
    python - "$IT" "$A_IN" "$C_IN" "$REPO" << "PY"
import sys
from huggingface_hub import HfApi
it, a_in, c_in, repo = sys.argv[1:5]
api = HfApi(token=open("/root/.hf_token").read().strip())
api.upload_folder(folder_path=a_in, repo_id=repo, path_in_repo=f"raw/iter_{it}/actor")
print(f"uploaded raw/iter_{it}/actor", flush=True)
api.upload_folder(folder_path=c_in, repo_id=repo, path_in_repo=f"raw/iter_{it}/critic")
print(f"uploaded raw/iter_{it}/critic", flush=True)
PY
fi

if [ "$RM" = "--rm" ]; then
    echo "=== freeing local iter_$IT ==="
    rm -rf "$A_IN" "$C_IN" "$A_OUT"
fi
df -h / | tail -1; echo "PUSH_CKPT_DONE $IT $MODE"
