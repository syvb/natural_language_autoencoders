#!/bin/bash
# Gate D0.3 generation, take 2: server + 2 staggered client shards in one
# supervised script. Resumable (gend_out jsonl). Avoids the pkill-cascade
# footgun: no teardown pkill here; the pod dies when the work is pulled.
set -uo pipefail
cd /work/gate_d

if ! curl -sf localhost:30000/health_generate > /dev/null 2>&1; then
  python -m sglang.launch_server --model-path models/av --port 30000 \
      --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
      > sglang.log 2>&1 &
  for i in $(seq 1 120); do
      curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo "sglang up"; break; }
      sleep 5
  done
fi
curl -sf localhost:30000/health_generate > /dev/null || { echo "SERVER_NOT_UP"; exit 1; }

SHARD="0/2" python gen_d.py > gend2_shard0.log 2>&1 &
P0=$!
sleep 30
SHARD="1/2" python gen_d.py > gend2_shard1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "GEN2_ALL_DONE"
