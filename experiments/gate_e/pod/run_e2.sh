#!/bin/bash
# Gate E2: train adapter (2 seeds) then eval generation through sglang.
set -uo pipefail
cd /work/gate_e
python train_adapter.py --seed 0 --out W_seed0.npz > train_W0.log 2>&1
echo DONE_W0
python train_adapter.py --seed 1 --out W_seed1.npz > train_W1.log 2>&1
echo DONE_W1
python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
    > sglang.log 2>&1 &
for i in $(seq 1 120); do
    curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo sglang_up; break; }
    sleep 5
done
curl -sf localhost:30000/health_generate > /dev/null || { echo SERVER_NOT_UP; exit 1; }
W_FILE=W_seed0.npz python gen_eval_adapter.py > evalW0.log 2>&1
echo DONE_EVAL0
W_FILE=W_seed1.npz python gen_eval_adapter.py > evalW1.log 2>&1
echo E2_ALL_DONE
