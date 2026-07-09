#!/bin/bash
# AR (critic) warm-start SL on Sonnet-4.6 ar-* data, continuing from the RLed critic
# (rl-ar is already a 20-layer model with value_head — no prepare_critic_checkpoint needed).
# Mirrors configs/critic_sft.sh (Qwen2.5 settings) adapted to a single H200/H100.
set -euo pipefail
cd /workspace/miles
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /root/.wandb_key)
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
    --prompt-data /workspace/out/ar_sft.parquet \
    --input-key prompt \
    --hf-checkpoint /workspace/models/rl-ar \
    --save /workspace/ckpt/ar \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 16 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --num-epoch 1 \
    --save-interval 500 \
    --use-wandb --wandb-project nla-warmstart-sonnet46 --wandb-team octahedral-systems \
    --wandb-mode online --wandb-group qwen2.5-7b-L20-ar \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation sdpa \
    "$@"
