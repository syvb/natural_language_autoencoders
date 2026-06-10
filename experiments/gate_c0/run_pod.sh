#!/bin/bash
set -euo pipefail
cd /root/gate_c0
pip install -q orjson pyyaml datasets accelerate 2>&1 | tail -1 || true
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir models/av 2>&1 | tail -1
hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir models/ar 2>&1 | tail -1
hf download Qwen/Qwen2.5-7B-Instruct 2>&1 | tail -1
echo "=== sglang launch ==="
python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
    > sglang.log 2>&1 &
SGLANG_PID=$!
for i in $(seq 1 120); do
    curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo "sglang up"; break; }
    kill -0 $SGLANG_PID 2>/dev/null || { echo "sglang died:"; tail -20 sglang.log; exit 1; }
    sleep 5
done
echo "=== generate (2x180x8) ==="
python bo_pod.py generate
echo "=== teardown sglang ==="
kill $SGLANG_PID 2>/dev/null || true
pkill -f sglang.launch_server 2>/dev/null || true
for i in $(seq 1 24); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$USED" -lt 5000 ] && break
    sleep 5
done
echo "=== score ==="
python bo_pod.py score
echo "=== select ==="
python select_candidates.py
echo "=== klpatch ==="
python bo_pod.py klpatch
echo "=== GATE C0 DONE ==="
