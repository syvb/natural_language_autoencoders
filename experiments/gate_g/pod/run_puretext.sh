#!/bin/bash
# Gate G pure-text pre-test: extract h24 -> build 3 text-only arms -> train 3
# text->vector ARs (--no-cond) -> reconstruct h24.
set -uo pipefail
cd /work/gate_g
export WANDB_API_KEY=$(cat ~/.wandb_key)

python extract_gap8.py > extract_pt.log 2>&1
echo EXTRACT_DONE
python build_puretext.py > build_pt.log 2>&1
echo BUILD_DONE
for arm in ptA ptC ptB; do
  python train_critic_cond.py --no-cond --pairs arm_${arm}_train.json \
    --eval-pairs arm_${arm}_eval.json --targets /work/gate_c/acts24.npy \
    --run-name pt_${arm} --out ckpt_${arm} > train_${arm}.log 2>&1
  echo "PT_DONE_${arm} cos=$(python3 -c "import json;print(round(json.load(open('ckpt_${arm}/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
done
echo PURETEXT_ALL_DONE >> retrain.log
