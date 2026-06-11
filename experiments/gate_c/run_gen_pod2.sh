#!/bin/bash
set -euo pipefail
cd /root/gate_c
pip install -q orjson pyyaml 2>&1 | tail -1 || true
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir models/av 2>&1 | tail -1
python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
    > sglang.log 2>&1 &
SGLANG_PID=$!
for i in $(seq 1 120); do
    curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo "sglang up"; break; }
    kill -0 $SGLANG_PID 2>/dev/null || { echo "sglang died:"; tail -20 sglang.log; exit 1; }
    sleep 5
done
for k in 8 9 10 11 12 13 14 15; do
  SHARD="$k/16" python gen2.py > gen2_shard$k.log 2>&1 &
done
wait
echo "GEN ALL SHARDS DONE"
