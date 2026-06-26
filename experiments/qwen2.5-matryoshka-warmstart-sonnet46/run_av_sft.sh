#!/bin/bash
# AV (actor) warm-start SFT on Sonnet-4.6 av-* data, continuing from the RLed actor.
# Mirrors configs/actor_sft.sh (Qwen2.5 settings) adapted to a single H200/H100.
# Run from the miles checkout root (where train_sft.py + train.py live).
set -euo pipefail
cd /workspace/miles
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /root/.wandb_key)
/opt/conda/bin/python train_sft.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --loss-type sft_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_actor.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data /workspace/out/av_sft.parquet \
    --input-key prompt \
    --hf-checkpoint /workspace/models/rl-av \
    --save /workspace/ckpt/av \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 16 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --loss-mask-type qwen \
    --nla-injection-scale 150 \
    --num-epoch 1 \
    --save-interval 500 \
    --use-wandb --wandb-project nla-warmstart-sonnet46 --wandb-team octahedral-systems \
    --wandb-mode online --wandb-group qwen2.5-7b-L20-av \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation flash_attention_2 \
    "$@"
