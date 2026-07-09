#!/bin/bash
# Preserve the run's non-checkpoint artifacts on HF: RL parquet (+ sidecar),
# launch configs, training log, archived rollout dumps.
# Usage: push_run_meta.sh [run_dir]
set -euo pipefail
RUN_DIR="${1:-/workspace/rl_quotepen}"
export HF_HUB_ENABLE_HF_TRANSFER=1
python - "$RUN_DIR" << "PY"
import glob, os, sys
from huggingface_hub import HfApi
run_dir = sys.argv[1]
repo = "syvb/nla-qwen2.5-7b-L20-rl-quotepen"
api = HfApi(token=open("/root/.hf_token").read().strip())
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
todo = []
rl = os.environ.get("RL_PARQUET", "/workspace/out/rl_tagged.parquet")
todo += [(rl, "data/rl_tagged.parquet"), (rl + ".nla_meta.yaml", "data/rl_tagged.parquet.nla_meta.yaml")]
todo += [(p, f"run/{os.path.basename(p)}") for p in glob.glob(f"{run_dir}/launch_config.*.txt")]
todo += [(p, f"run/{os.path.basename(p)}") for p in glob.glob("/workspace/out/train*.log")]
todo += [(p, f"run/dumps/{os.path.basename(p)}") for p in glob.glob("/workspace/out/dumps/*.txt")]
for src, dst in todo:
    if not os.path.exists(src):
        print("skip (missing):", src); continue
    print("upload", src, "->", dst, flush=True)
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo)
print("PUSH_META_DONE")
PY
