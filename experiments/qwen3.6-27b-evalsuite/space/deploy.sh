#!/bin/bash
# Deploy the NLA Qwen3.6-27B explorer Space to HF.
#
# The Space is self-contained: app.py + the vendored ./nla runtime subset +
# nla_meta.yaml + mu.npy + default_texts.json. No precache (default-text clicks
# compute live). Needs ~/.hf_token.
#
#   SPACE_REPO=you/nla-qwen36-27b-explorer bash deploy.sh
#
# Deploys to ZeroGPU (zero-a10g). The app itself requests the full RTX Pro 6000
# Blackwell (96GB) via @spaces.GPU(size="xlarge") — needed because the ~93GB
# actor+critic exceed the 48GB "large" default slice. ZeroGPU needs a PRO/Team
# account on the SPACE_REPO owner. See README "Hardware".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SPACE_REPO="${SPACE_REPO:?set SPACE_REPO=owner/space-name}"
HARDWARE="${HARDWARE:-zero-a10g}"
BUILD="${BUILD_DIR:-/tmp/nla_space27_build}"

echo "=== stage build dir $BUILD ==="
rm -rf "$BUILD"; mkdir -p "$BUILD"
cp "$HERE"/app.py "$HERE"/README.md "$HERE"/requirements.txt \
   "$HERE"/nla_meta.yaml "$HERE"/mu.npy "$HERE"/default_texts.json "$BUILD/"
cp -r "$HERE/nla" "$BUILD/nla"   # vendored EasyNLA runtime subset

echo "=== sanity: app.py compiles, vendored nla imports ==="
"$PY" -m py_compile "$BUILD/app.py"
( cd "$BUILD" && "$PY" -c "import nla.config, nla.models, nla.utils, nla.injection; print('nla OK')" )

# PRIVATE=1 by default: the 27B runtime (ZeroGPU 96GB fit, CPU-RAM staging)
# can't be verified offline, so ship private and flip to public once it runs.
PRIVATE="${PRIVATE:-1}"
echo "=== create + upload Space ($HARDWARE, private=$PRIVATE) ==="
# BUCKET: an HF Bucket holds the ~92GB bf16 weight cache (HF_HOME=/bucket/hf in
# app.py), off the Space's 150GB ephemeral limit. Created + mounted rw here.
BUCKET="${BUCKET:-syvb/nla-qwen36-27b-cache}"
"$PY" - "$SPACE_REPO" "$BUILD" "$HARDWARE" "$PRIVATE" "$BUCKET" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi, create_bucket, Volume
repo, build, hardware, private, bucket = sys.argv[1:6]
tok = open(os.path.expanduser("~/.hf_token")).read().strip()
api = HfApi(token=tok)
api.create_repo(repo, repo_type="space", space_sdk="gradio",
                space_hardware=hardware, private=private == "1", exist_ok=True)
create_bucket(bucket, exist_ok=True, token=tok)
api.set_space_volumes(repo, volumes=[
    Volume(type="bucket", source=bucket, mount_path="/bucket", read_only=False)])
api.upload_folder(folder_path=build, repo_id=repo, repo_type="space",
                  ignore_patterns=["__pycache__/*", "*.pyc"])
print(f"deployed https://huggingface.co/spaces/{repo} (bucket {bucket} at /bucket)")
PYEOF
