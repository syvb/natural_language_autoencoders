#!/bin/bash
# Push ONE RL checkpoint (actor + critic) to HF in usable-model form.
# Keeps the local iter dirs (they are what makes the run resumable).
#
# Usage:  push_ckpt_to_hf.sh <iter_padded> [run_dir]
#   e.g.  push_ckpt_to_hf.sh 0000050
# Needs HF_TOKEN (or HF_TOKEN_FILE), the warm-start AV HF dir as the actor
# architecture origin (ACTOR_ORIGIN, default $WORK/hf_out/av_ws), and nla
# importable. Repo: ${HF_REPO_PREFIX}-rl / iter_<N>/{av,ar}.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_config.sh"
IT="${1:?usage: push_ckpt_to_hf.sh <iter_padded> [run_dir]}"
RUN_DIR="${2:-$WORK/rl_run}"
if [ -z "${HF_TOKEN:-}" ]; then
    [ -f "$HF_TOKEN_FILE" ] || { echo "FATAL: $HF_TOKEN_FILE missing" >&2; exit 1; }
    HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
REPO="${HF_REPO_PREFIX}-rl"
A_IN="$RUN_DIR/actor/iter_$IT"
C_HF="$RUN_DIR/critic/iter_$IT/hf"
A_OUT="$WORK/hf_out/rl_$IT"
ACTOR_ORIGIN="${ACTOR_ORIGIN:-$WORK/hf_out/av_ws}"

echo "=== verify critic value_head is finite ==="
"${PYTHON:-python}" - "$C_HF/value_head.safetensors" << "PY"
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

echo "=== convert actor DCP -> HF ==="
rm -rf "$A_OUT"
"${PYTHON:-python}" "$HERE/../../tools/convert_fsdp_to_hf.py" \
    --input-dir "$A_IN" --output-dir "$A_OUT" \
    --origin-hf-dir "$ACTOR_ORIGIN" -f
cp "$A_IN/nla_meta.yaml" "$A_OUT/nla_meta.yaml"

if [ "${UPLOAD:-1}" != "1" ]; then
    echo "[UPLOAD=0] export verified + converted: $A_OUT and $C_HF (skipping HF upload)"
    echo "PUSH_DONE iter_$IT (local only)"
    exit 0
fi
echo "=== upload iter_$IT to $REPO ==="
"${PYTHON:-python}" - "$A_OUT" "$C_HF" "$REPO" "$IT" << "PY"
import sys
from huggingface_hub import HfApi
a_out, c_hf, repo, it = sys.argv[1:5]
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=a_out, repo_id=repo, path_in_repo=f"iter_{it}/av")
api.upload_folder(folder_path=c_hf, repo_id=repo, path_in_repo=f"iter_{it}/ar")
print(f"PUSHED https://huggingface.co/{repo}/tree/main/iter_{it}")
PY
echo "PUSH_DONE iter_$IT"
