#!/bin/bash
# Step 4b — AR (critic) warm-start SL, FROM SCRATCH (init = 03's truncated
# base + identity value head). Trains on the token-TRUNCATED ar_sft parquet
# (K ~ U[NLA_TRUNC_MIN_TOKENS, NLA_TRUNC_MAX_TOKENS], matching RL's draw) so
# the critic is pre-calibrated for the short prefixes it meets at RL step 0.
# FSDP over SFT_GPUS. Independent of run_av_sft.sh — run both in parallel on
# separate GPU sets/boxes if wall-clock matters.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_config.sh"

export MILES_DIR="${MILES_DIR:-$WORK/miles}"
cd "$MILES_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [ -z "${WANDB_API_KEY:-}" ]; then
    [ -f "$WANDB_KEY_FILE" ] || { echo "FATAL: $WANDB_KEY_FILE missing (wandb is mandatory)" >&2; exit 1; }
    WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi
export WANDB_API_KEY
AR_PARQUET="${AR_PARQUET:-$WORK/out/ar_sft.parquet}"

"${PYTHON:-python}" "$HERE/train_sft.py" \
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
    --hf-checkpoint "${AR_HF_CKPT:-$WORK/models/critic_init}" \
    --save "${AR_SAVE:-$WORK/ckpt/ar_ws}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$SFT_GPUS" \
    --rollout-batch-size "$SFT_GLOBAL_BS" \
    --global-batch-size "$SFT_GLOBAL_BS" \
    --micro-batch-size "$SFT_MICRO_BS" \
    --lr "$SFT_LR" --min-lr "$SFT_MIN_LR" --lr-warmup-iters "$SFT_WARMUP" --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --num-epoch "$SFT_EPOCHS" \
    --save-interval 500 \
    --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-team "$WANDB_TEAM" \
    --wandb-mode online --wandb-group "${WANDB_GROUP:-${MODEL_TAG}-ar-ws}" \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation sdpa \
    "$@"
