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
# precache.json (made by precompute_cache.py) — instant default-text clicks
[[ -f "$HERE/precache.json" ]] && cp "$HERE/precache.json" "$BUILD/" \
  && echo "  + precache.json ($(wc -c < "$HERE/precache.json") bytes)" \
  || echo "  (no precache.json — default-text clicks compute live)"
# eval_awareness.json (made by eval_awareness.py) — the heatmap-toggle overlay
[[ -f "$HERE/eval_awareness.json" ]] && cp "$HERE/eval_awareness.json" "$BUILD/" \
  && echo "  + eval_awareness.json"
# loo.json (made by loo_precompute.py) — per-line ablation (necessity) scores
[[ -f "$HERE/loo.json" ]] && cp "$HERE/loo.json" "$BUILD/" \
  && echo "  + loo.json"

echo "=== sanity: app.py compiles, vendored nla imports ==="
"$PY" -m py_compile "$BUILD/app.py"
( cd "$BUILD" && "$PY" -c "import nla.config, nla.models, nla.utils, nla.injection; print('nla OK')" )

# PRIVATE=1 by default: the 27B runtime (ZeroGPU 96GB fit, CPU-RAM staging)
# can't be verified offline, so ship private and flip to public once it runs.
PRIVATE="${PRIVATE:-1}"
echo "=== create + upload Space ($HARDWARE, private=$PRIVATE) ==="
"$PY" - "$SPACE_REPO" "$BUILD" "$HARDWARE" "$PRIVATE" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi
repo, build, hardware, private = sys.argv[1:5]
tok = open(os.path.expanduser("~/.hf_token")).read().strip()
api = HfApi(token=tok)
api.create_repo(repo, repo_type="space", space_sdk="gradio",
                space_hardware=hardware, private=private == "1", exist_ok=True)
api.upload_folder(folder_path=build, repo_id=repo, repo_type="space",
                  ignore_patterns=["__pycache__/*", "*.pyc"])
print(f"deployed https://huggingface.co/spaces/{repo}")
PYEOF
