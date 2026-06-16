#!/bin/bash
# Gate G Stage 0 pipeline: AV verbalizes h_22 -> build arms -> train 3
# h_20-conditioned critics (none / cat / text). Run AFTER a manual smoke check.
set -uo pipefail
cd /work/gate_g

python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.85 --context-length 1024 \
    > sglang.log 2>&1 &
for i in $(seq 1 120); do
  curl -sf localhost:30000/health_generate >/dev/null 2>&1 && { echo sglang_up; break; }
  sleep 5
done
curl -sf localhost:30000/health_generate >/dev/null || { echo SERVER_NOT_UP; exit 1; }

python gen_av.py > gen_av.log 2>&1
echo GEN_DONE
pkill -f 'sglang.launch_serve[r]' 2>/dev/null || true
sleep 8

python build_arms_g.py > build.log 2>&1
echo BUILD_DONE

for arm in none cat text; do
  python train_critic_cond.py --pairs arm_${arm}_train.json \
     --eval-pairs arm_${arm}_eval.json --run-name g_${arm} --out ckpt_${arm} \
     > train_${arm}.log 2>&1
  c=$(python3 -c "import json;print(json.load(open('ckpt_${arm}/RESULT.json'))['final_eval_cos'])" 2>/dev/null)
  echo "TRAIN_DONE_${arm} cos=${c}"
done
echo S0_ALL_DONE
