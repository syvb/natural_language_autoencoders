#!/bin/bash
# AR (critic) v2 warm-start SL — on the item-TRUNCATED, list-format ar_sft parquet.
# Same as run_ar_sft.sh but --prompt-data the v2 parquet (built by
# 02b_build_datasets_v2.py with --ar-truncate-max-items). The truncation is baked
# into the data (one taper-drawn K per row; warm-start is a single epoch), so the
# critic SFT itself is unchanged — it just sees short prefixes now, pre-calibrating
# it for RL step 0 (v2 item 2). No NLA_NO_TRAIN_EOS here (critic uses last-token MSE,
# no generation).
set -euo pipefail
cd /workspace/miles
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /root/.wandb_key)
AR_PARQUET="${AR_PARQUET:-/workspace/out/ar_sft_v2.parquet}"
/opt/conda/bin/python train_sft.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --nla-model-is-critic \
    --loss-type custom_loss \
    --custom-loss-function-path nla.loss.nla_critic_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_critic.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$AR_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "${AR_HF_CKPT:-/workspace/models/rl-ar}" \
    --save "${AR_SAVE:-/workspace/ckpt/ar_v2}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 16 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --num-epoch 1 \
    --save-interval 500 \
    --use-wandb --wandb-project nla-warmstart-sonnet46-v2 --wandb-team octahedral-systems \
    --wandb-mode online --wandb-group qwen2.5-7b-L20-ar-v2 \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation sdpa \
    "$@"
