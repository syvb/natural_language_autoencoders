#!/bin/bash
set -eo pipefail
echo "===== DOWNLOAD START $(date) ====="
/opt/conda/bin/pip install -q "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/models
/opt/conda/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or None
jobs = [
    ("Qwen/Qwen2.5-7B-Instruct", "/workspace/models/base"),
    ("kitft/nla-qwen2.5-7b-L20-av", "/workspace/models/av"),
    ("kitft/nla-qwen2.5-7b-L20-ar", "/workspace/models/ar"),
]
for repo, dst in jobs:
    print("== downloading", repo, "->", dst, flush=True)
    snapshot_download(repo_id=repo, local_dir=dst, token=tok,
                      ignore_patterns=["*.pth", "original/*", "*.gguf"])
    print("== done", repo, flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)
PY
echo "===== DOWNLOAD DONE $(date) ====="
du -sh /workspace/models/* 2>/dev/null
