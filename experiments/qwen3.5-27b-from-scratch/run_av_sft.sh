#!/bin/bash
# Step 4a — AV (actor) warm-start SFT, FROM SCRATCH (init = the plain base
# model; no prior NLA checkpoint exists for this model). Bullets format +
# never-train-EOS, FSDP over SFT_GPUS.
#
# INJECTION_SCALE is required — measure it with 01 (norm_stats.json), never
# guess. --nla-sidecar-source points at the training parquet so the exported
# checkpoint records the template it was actually trained on.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_config.sh"
: "${INJECTION_SCALE:?measure with 01_extract_activations.py (norm_stats.json) and set in config}"

export MILES_DIR="${MILES_DIR:-$WORK/miles}"
cd "$MILES_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [ -z "${WANDB_API_KEY:-}" ]; then
    [ -f "$WANDB_KEY_FILE" ] || { echo "FATAL: $WANDB_KEY_FILE missing (wandb is mandatory)" >&2; exit 1; }
    WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi
export WANDB_API_KEY
export NLA_NO_TRAIN_EOS=1
AV_PARQUET="${AV_PARQUET:-$WORK/out/av_sft.parquet}"

"${PYTHON:-python}" "$HERE/train_sft.py" \
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
    --hf-checkpoint "${AV_HF_CKPT:-$WORK/models/base}" \
    --save "${AV_SAVE:-$WORK/ckpt/av_ws}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$SFT_GPUS" \
    --rollout-batch-size "$SFT_GLOBAL_BS" \
    --global-batch-size "$SFT_GLOBAL_BS" \
    --micro-batch-size "$SFT_MICRO_BS" \
    --lr "$SFT_LR" --min-lr "$SFT_MIN_LR" --lr-warmup-iters "$SFT_WARMUP" --lr-decay-style cosine \
    --n-samples-per-prompt 1 \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    --nla-injection-scale "$INJECTION_SCALE" \
    --num-epoch "$SFT_EPOCHS" \
    --save-interval 500 \
    --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-team "$WANDB_TEAM" \
    --wandb-mode online --wandb-group "${WANDB_GROUP:-${MODEL_TAG}-av-ws}" \
    --sglang-router-ip 127.0.0.1 --sglang-router-port 39999 \
    --attn-implementation "$ATTN_IMPL" \
    "$@"
