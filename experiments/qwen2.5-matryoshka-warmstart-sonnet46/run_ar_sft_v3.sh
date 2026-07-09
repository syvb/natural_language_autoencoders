#!/bin/bash
# AR (critic) v3 warm-start SL — on the token-TRUNCATED, bullets-format ar_sft
# parquet. Inits from the BASE kitft NLA critic (kitft/nla-qwen2.5-7b-L20-ar,
# ships value_head.safetensors + nla_meta.yaml), NOT the v1/v2 checkpoints.
# Pre-download it to $AR_HF_CKPT (default below).
# Same as run_ar_sft_v2.sh but --prompt-data the v3 parquet (built by
# 02c_build_datasets_v3.py with --ar-truncate-max-tokens: K ~ U[1, 120] token
# prefixes, matching RL's uniform tokens-mode truncation). The truncation is
# baked into the data (one draw per row; warm-start is a single epoch), so the
# critic SFT itself is unchanged — it just sees short prefixes now, down to a
# single token, pre-calibrating it for RL step 0. No NLA_NO_TRAIN_EOS here
# (critic uses last-token MSE, no generation).
set -euo pipefail
cd /workspace/miles
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /root/.wandb_key)
AR_PARQUET="${AR_PARQUET:-/workspace/out/ar_sft_v3.parquet}"
# --nla-sidecar-source: same rationale as run_av_sft_v3.sh — don't inherit the
# kitft base checkpoint's stale (tagged-template) sidecar into the v3 export.
/opt/conda/bin/python train_sft.py \
    --train-backend fsdp \
    --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
    --nla-sidecar-source "$AR_PARQUET" \
    --nla-model-is-critic \
    --loss-type custom_loss \
    --custom-loss-function-path nla.loss.nla_critic_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_critic.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$AR_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "${AR_HF_CKPT:-/workspace/models/nla-ar}" \
    --save "${AR_SAVE:-/workspace/ckpt/ar_v3}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-batch-size 256 \
    --global-batch-size 256 \
    --micro-batch-size 16 \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 50 --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --num-epoch 1 \
    --save-interval 500 \
    --use-wandb --wandb-project nla-warmstart-sonnet46-v3 --wandb-team octahedral-systems \
    --wandb-mode online --wandb-group qwen2.5-7b-L20-ar-v3 \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation sdpa \
    "$@"
