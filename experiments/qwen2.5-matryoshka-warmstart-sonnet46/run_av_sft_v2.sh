#!/bin/bash
# AV (actor) v2 warm-start SFT — list format + never-train-EOS.
# Inits from the BASE kitft NLA actor (kitft/nla-qwen2.5-7b-L20-av), NOT the v1
# truncation/matryoshka checkpoints — v2 is a fresh take on truncation and should
# not inherit v1's behavior. Pre-download it to $AV_HF_CKPT (default below).
# Same as run_av_sft.sh but:
#   - NLA_NO_TRAIN_EOS=1 (item 5): loss is masked at the turn terminator so the
#     actor is never taught to stop — generalizes to arbitrarily long output.
#   - --prompt-data the v2 (list-format, unwrapped) av_sft parquet.
# Targets are the raw newline list (no <explanation> wrapper); the loss-mask-type
# is still qwen (chat scaffolding), only the EOS position is zeroed.
set -euo pipefail
cd /workspace/miles
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /root/.wandb_key)
export NLA_NO_TRAIN_EOS=1
AV_PARQUET="${AV_PARQUET:-/workspace/out/av_sft_v2.parquet}"
/opt/conda/bin/python train_sft.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --loss-type sft_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_actor.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$AV_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "${AV_HF_CKPT:-/workspace/models/nla-av}" \
    --save "${AV_SAVE:-/workspace/ckpt/av_v2}" \
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
    --use-wandb --wandb-project nla-warmstart-sonnet46-v2 --wandb-team octahedral-systems \
    --wandb-mode online --wandb-group qwen2.5-7b-L20-av-v2 \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation flash_attention_2 \
    "$@"
