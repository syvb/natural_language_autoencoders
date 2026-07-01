#!/bin/bash
# AV (actor) v3 warm-start SFT — bullets format + never-train-EOS.
# Inits from the BASE kitft NLA actor (kitft/nla-qwen2.5-7b-L20-av), NOT the
# v1/v2 checkpoints — v3 is a fresh take and should not inherit their behavior.
# Pre-download it to $AV_HF_CKPT (default below).
# Same as run_av_sft_v2.sh but --prompt-data the v3 (bullets-format) av_sft
# parquet from 02c_build_datasets_v3.py. Targets are a raw "- " bullet list
# (no <explanation> wrapper); NLA_NO_TRAIN_EOS=1 masks loss at the turn
# terminator so the actor is never taught to stop.
set -euo pipefail
cd /workspace/miles
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /root/.wandb_key)
export NLA_NO_TRAIN_EOS=1
AV_PARQUET="${AV_PARQUET:-/workspace/out/av_sft_v3.parquet}"
# --nla-sidecar-source: WITHOUT it, resolve_sidecar_source prefers the kitft
# base checkpoint's nla_meta.yaml (v1 TAGGED template) over the parquet's, and
# _write_sidecar bakes that stale template into every exported v3 checkpoint —
# which then feeds RL's opening-offset detection and ships in the released
# model (the v2 sidecar bug, second leg). Point it at the training parquet so
# the checkpoint records what the model was actually trained on.
/opt/conda/bin/python train_sft.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --nla-sidecar-source "$AV_PARQUET" \
    --loss-type sft_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_actor.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$AV_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "${AV_HF_CKPT:-/workspace/models/nla-av}" \
    --save "${AV_SAVE:-/workspace/ckpt/av_v3}" \
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
    --use-wandb --wandb-project nla-warmstart-sonnet46-v3 --wandb-team octahedral-systems \
    --wandb-mode online --wandb-group qwen2.5-7b-L20-av-v3 \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation flash_attention_2 \
    "$@"
