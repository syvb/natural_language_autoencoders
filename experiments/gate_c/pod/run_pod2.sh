#!/bin/bash
# Pod 2: armA_8k then armB_8k. KL judge runs as a separate step
# (kl_judge.py) once preds_arm0p_8k arrives from pod 1.
cd /work/gate_c
for arm in "armA_8k armA_train.json armA_eval.json" \
           "armB_8k armB_train.json armB_eval.json"; do
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
echo POD2_DONE
