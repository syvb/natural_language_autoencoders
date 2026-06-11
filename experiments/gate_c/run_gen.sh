#!/bin/bash
set -uo pipefail
cd /root/gate_c
for k in 0 1 2 3; do
  SHARD="$k/4" python gen2.py > gen2_shard$k.log 2>&1 &
done
wait
grep -h "ALL DONE" gen2_shard*.log | head -1 && echo "GEN ALL SHARDS DONE"
