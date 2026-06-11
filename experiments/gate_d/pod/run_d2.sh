#!/bin/bash
# Gate D2: train 4 critic arms on attention-write targets + export preds.
# Expects /work/gate_d populated: train_critic.py, predict_critic.py,
# nla_inference.py, attn_out.npy, arm{C,T,H,Z}_{train8k,eval8k}.json,
# models/ar, ~/.wandb_key.
set -uo pipefail
cd /work/gate_d
for arm in armC armT armH armZ; do
  name="${arm}_8k"
  python train_critic.py --pairs ${arm}_train8k.json --targets attn_out.npy \
    --eval-pairs ${arm}_eval8k.json --eval-targets attn_out.npy \
    --out ckpt_$name --run-name $name > train_$name.log 2>&1
  cp models/ar/nla_meta.yaml ckpt_$name/
  python predict_critic.py --ckpt ckpt_$name --pairs ${arm}_eval8k.json \
    --name $name > pred_$name.log 2>&1
  echo "DONE_$name"
done
echo D2_ALL_DONE
