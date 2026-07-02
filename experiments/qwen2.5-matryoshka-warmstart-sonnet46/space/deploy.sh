#!/bin/bash
# Assemble + deploy the NLA v3 explorer Space to HF (syvb/nla-v3-explorer).
#
# nla_inference.py is NOT committed here (it's the repo-root copy — committing a
# duplicate would drift). This script stages it alongside the Space source into
# a build dir, regenerates the data assets, and uploads. Needs ~/.hf_token.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-python3}"
SPACE_REPO="syvb/nla-v3-explorer"
BUILD="${BUILD_DIR:-/tmp/nla_space_build}"

echo "=== regenerate data assets ==="
"$PY" "$HERE/build_assets.py"

echo "=== stage build dir $BUILD ==="
rm -rf "$BUILD"; mkdir -p "$BUILD"
cp "$HERE"/app.py "$HERE"/README.md "$HERE"/requirements.txt \
   "$HERE"/mu.npy "$HERE"/default_texts.json "$BUILD/"
cp "$REPO_ROOT/nla_inference.py" "$BUILD/nla_inference.py"   # vendored at deploy, not committed

# precache.json: baked per-position results for the default texts (instant
# clicks, no GPU). Optional — the Space works without it, just slower.
if [[ -f "$HERE/precache.json" ]]; then
  cp "$HERE/precache.json" "$BUILD/"
  "$PY" - "$HERE" <<'PYEOF'
import json, sys
here = sys.argv[1]
texts = json.load(open(f"{here}/default_texts.json"))
cached = {e["text"] for e in json.load(open(f"{here}/precache.json"))["entries"]}
missing = [t for t in texts if t not in cached]
if missing:
    print(f"WARNING: precache.json is STALE — {len(missing)}/{len(texts)} default "
          f"texts uncovered (clicks on them compute live).")
    print("  Regenerate with: bash precompute_on_vast.sh")
PYEOF
else
  echo "WARNING: no precache.json — default-text clicks will compute live."
  echo "  Generate with: bash precompute_on_vast.sh"
fi

echo "=== create + upload Space (zero-a10g, public) ==="
"$PY" - "$SPACE_REPO" "$BUILD" <<'PYEOF'
import sys
from huggingface_hub import HfApi
repo, build = sys.argv[1], sys.argv[2]
api = HfApi(token=open(__import__("os").path.expanduser("~/.hf_token")).read().strip())
api.create_repo(repo, repo_type="space", space_sdk="gradio",
                space_hardware="zero-a10g", private=False, exist_ok=True)
api.upload_folder(folder_path=build, repo_id=repo, repo_type="space",
                  ignore_patterns=["__pycache__/*", "*.pyc"])
print(f"deployed https://huggingface.co/spaces/{repo}")
PYEOF
