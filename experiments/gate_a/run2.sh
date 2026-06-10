#!/bin/bash
set -euo pipefail
cd /root/gate_a
echo "=== sglang launch ==="
python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
    > sglang.log 2>&1 &
SGLANG_PID=$!
for i in $(seq 1 120); do
    curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo "sglang up after ${i}x5s"; break; }
    kill -0 $SGLANG_PID 2>/dev/null || { echo "sglang died:"; tail -30 sglang.log; exit 1; }
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
