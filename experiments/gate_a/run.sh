#!/bin/bash
# Gate A driver — runs on the pod from /root/gate_a
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

echo "=== deps ==="
pip install -q datasets orjson pyyaml httpx 2>&1 | tail -1 || true

echo "=== downloads ==="
huggingface-cli download kitft/nla-qwen2.5-7b-L20-av --local-dir models/av 2>&1 | tail -1
huggingface-cli download kitft/nla-qwen2.5-7b-L20-ar --local-dir models/ar 2>&1 | tail -1
huggingface-cli download Qwen/Qwen2.5-7B-Instruct 2>&1 | tail -1

echo "=== extract ==="
python gate_a.py extract

echo "=== sglang launch ==="
python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
    > sglang.log 2>&1 &
SGLANG_PID=$!
for i in $(seq 1 120); do
    if curl -sf localhost:30000/health_generate > /dev/null 2>&1; then
        echo "sglang up after ${i}x5s"; break
    fi
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "sglang died:"; tail -30 sglang.log; exit 1
    fi
    sleep 5
done
curl -sf localhost:30000/health_generate > /dev/null || { echo "sglang health timeout"; tail -30 sglang.log; exit 1; }

echo "=== generate ==="
python gate_a.py generate

echo "=== teardown sglang ==="
kill $SGLANG_PID 2>/dev/null || true
pkill -f sglang.launch_server 2>/dev/null || true
for i in $(seq 1 24); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$USED" -lt 5000 ] && break
    sleep 5
done
echo "gpu mem now: ${USED}MB"

echo "=== score ==="
python gate_a.py score

echo "=== GATE A DONE ==="
