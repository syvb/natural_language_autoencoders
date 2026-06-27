#!/bin/bash
# Push ONE RL checkpoint (actor + critic) to HuggingFace and free local disk.
# Converts the actor DCP -> HF (tools/convert_fsdp_to_hf.py), uploads actor (av/)
# and critic (ar/, already HF with value_head) to a per-iter subfolder, then
# deletes the local iter dir. Usable-model form (weights + value_head), NOT the
# optimizer state — so not resumable, but durable + eval-ready, and it keeps a
# small disk from filling during a long run.
#
# Usage:  push_checkpoint_to_hf.sh <iter_padded> [run_dir] [repo_prefix]
#   e.g.  push_checkpoint_to_hf.sh 0000200 /workspace/rl_kl kl0.01/
# Edit REPO below for a different target repo. Needs /root/.hf_token,
# /workspace/models/av as the actor architecture origin, and the nla pkg on path.
set -e
IT="$1"; RUN_DIR="${2:-/workspace/rl_kl}"; PREFIX="${3:-kl0.01/}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
REPO="syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard"
A_IN="$RUN_DIR/actor/iter_$IT"; C_HF="$RUN_DIR/critic/iter_$IT/hf"; A_OUT="/workspace/hf_out/$PREFIX$IT"
# Gate: never upload/delete a corrupt critic. The exported value_head.safetensors
# was historically half-garbage from an FSDP gather bug (now fixed at save time in
# nla/train_actor.save_model); this is the belt-and-suspenders postflight. With
# `set -e`, a non-finite head aborts BEFORE convert/upload/delete, preserving the
# local checkpoint for inspection.
echo "=== verify critic value_head is finite ==="
python - "$C_HF/value_head.safetensors" << "PY"
import sys, torch
from safetensors.torch import load_file
w = load_file(sys.argv[1])["weight"]
if not torch.isfinite(w).all():
    n = (~torch.isfinite(w)).sum().item()
    raise SystemExit(f"ABORT: value_head has {n} non-finite entries ({sys.argv[1]}); "
                     "NOT uploading or deleting. Re-export with the fixed save path.")
print("value_head finite OK")
PY
echo "=== convert actor DCP->HF ($PREFIX$IT) ==="
rm -rf "$A_OUT"; python /workspace/nla/tools/convert_fsdp_to_hf.py --input-dir "$A_IN" --output-dir "$A_OUT" --origin-hf-dir /workspace/models/av -f
[ -f "$A_IN/nla_meta.yaml" ] && cp "$A_IN/nla_meta.yaml" "$A_OUT/nla_meta.yaml"
python - "$IT" "$A_OUT" "$C_HF" "$REPO" "$PREFIX" << "PY"
import sys
from huggingface_hub import HfApi
it,a_out,c_hf,repo,prefix=sys.argv[1:6]
api=HfApi(token=open("/root/.hf_token").read().strip())
api.create_repo(repo,repo_type="model",private=True,exist_ok=True)
api.upload_folder(folder_path=a_out,repo_id=repo,repo_type="model",path_in_repo=f"{prefix}iter_{it}/av")
print(f"uploaded {prefix}iter_{it}/av",flush=True)
api.upload_folder(folder_path=c_hf,repo_id=repo,repo_type="model",path_in_repo=f"{prefix}iter_{it}/ar")
print(f"uploaded {prefix}iter_{it}/ar",flush=True)
PY
rm -rf "$RUN_DIR/actor/iter_$IT" "$RUN_DIR/critic/iter_$IT" "$A_OUT"
df -h / | tail -1; echo "PUSH_CKPT_DONE $IT"
