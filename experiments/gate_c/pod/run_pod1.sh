#!/bin/bash
# Pod 1: arm0p_8k (primary baseline) then armA_full (data-size slope).
cd /work/gate_c
for arm in "arm0p_8k arm0p_train.json arm0p_eval.json" \
           "armA_full armA_full_train.json armA_eval.json"; do
  set -- $arm
  name=$1; train=$2; eval=$3
  python train_critic.py --pairs $train --targets diff_targets.npy \
    --eval-pairs $eval --eval-targets diff_targets.npy \
    --out ckpt_$name --run-name $name > train_$name.log 2>&1
  cp models/ar/nla_meta.yaml ckpt_$name/
  python predict_critic.py --ckpt ckpt_$name --pairs $eval --name $name \
    > pred_$name.log 2>&1
  echo "DONE_$name"
done
echo POD1_DONE
