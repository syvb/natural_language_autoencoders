#!/bin/bash
# Gate D0.3 generation. Two gated steps:
#   bash run_gen_d.sh smoke  — launch sglang AV server, run 50-pos CJK smoke
#   bash run_gen_d.sh full   — (server already up) full gen, 4 client shards
set -uo pipefail
cd /work/gate_d

if [ "$1" = "smoke" ]; then
  python -m sglang.launch_server --model-path models/av --port 30000 \
      --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
      > sglang.log 2>&1 &
  SGLANG_PID=$!
  for i in $(seq 1 120); do
      curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo "sglang up"; break; }
      kill -0 $SGLANG_PID 2>/dev/null || { echo "SGLANG_DIED"; tail -20 sglang.log; exit 1; }
      sleep 5
  done
  python gen_d.py smoke
  echo "SMOKE_DONE"
elif [ "$1" = "full" ]; then
  for k in 0 1 2 3; do
    SHARD="$k/4" python gen_d.py > gend_shard$k.log 2>&1 &
  done
  wait
  pkill -f sglang.launch_server 2>/dev/null || true
  echo "GEN_ALL_DONE"
fi
