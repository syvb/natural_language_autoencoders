#!/bin/bash
# Gate D0 pod bootstrap: deps, models, extraction. Generation is launched
# separately after the local D0.2 ridge gate passes.
set -uo pipefail
cd /work/gate_d
export HF_TOKEN=$(cat /work/gate_d/.hf_token)
pip install -q datasets accelerate orjson pyyaml 2>&1 | tail -1 || true
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir models/av 2>&1 | tail -1
hf download Qwen/Qwen2.5-7B-Instruct 2>&1 | tail -1
echo "=== extract_d ==="
python extract_d.py
echo "EXTRACT_DONE"
