#!/bin/bash
# Gate E3: train reader (LoRA+W, 2 seeds), serve each merged model, eval.
set -uo pipefail
cd /work/gate_e
for s in 0 1; do
  python train_reader.py --seed $s --out-w W_e3_seed$s.npz --out-model av_e3_s$s \
    > train_e3_s$s.log 2>&1
  echo DONE_TRAIN_$s
done
for s in 0 1; do
  pkill -f 'sglang.launch_serve[r]' 2>/dev/null || true
  sleep 8
  python -m sglang.launch_server --model-path av_e3_s$s --port 30000 \
      --disable-radix-cache --mem-fraction-static 0.8 --context-length 1024 \
      > sglang_s$s.log 2>&1 &
  for i in $(seq 1 120); do
      curl -sf localhost:30000/health_generate > /dev/null 2>&1 && { echo sglang_up_$s; break; }
      sleep 5
  done
  curl -sf localhost:30000/health_generate > /dev/null || { echo SERVER_NOT_UP_$s; exit 1; }
  W_FILE=W_e3_seed$s.npz python gen_eval_adapter.py > eval_e3_s$s.log 2>&1
  echo DONE_EVAL_$s
done
echo E3_ALL_DONE
