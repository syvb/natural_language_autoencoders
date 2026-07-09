#!/bin/bash
# RL the v2 matryoshka Qwen2.5-7B L20 AV/AR pair. Thin wrapper over configs/rl.sh,
# same shape as run_rl_truncated.sh (v1) but with the v2 plan's changes:
#
#   1. Higher KL penalty (KL_LOSS_COEF default 0.02 = 2x v1's 0.01) — anchors the
#      AV harder to the warm-start reference (entropy/CJK collapse defense).
#   4. ITEM-mode random truncation (NLA_TRUNC_MODE=items): keep K newline list
#      items, K ~ taper over [1, MAX_ITEMS], shared per group, optionally annealed
#      long→short by a curriculum. Generation runs to the hard cap; nla_generate
#      post-truncates the rollout at the K-th newline.
#   4a. Per-item length penalty (NLA_ITEM_LEN_PENALTY) so the AV can't cram every-
#       thing into one giant item.
#
# These pair with the v2 warm-start (build datasets with --explanation-format list
# + --ar-truncate-max-items; warm-start the AV with NLA_NO_TRAIN_EOS=1). The AV
# emits a raw newline list (no <explanation> wrapper, never trained to stop), so
# extract_explanation_open reads the whole output.
#
# DEFAULT PROFILE: matches run_rl_truncated.sh — one 8×H100 node (actor 4 /
# critic 2 / rollout 2), 512-sample batch (64×8). Set NUM_ROLLOUT small for a smoke.
set -euo pipefail

# ───────────────────────── truncation: ITEM mode (v2) ──────────────────────
export NLA_TRUNC_MODE="${NLA_TRUNC_MODE:-items}"
export NLA_TRUNC_MAX_ITEMS="${NLA_TRUNC_MAX_ITEMS:-10}"     # keep K ~ U[1, this]
export NLA_TRUNC_TAPER="${NLA_TRUNC_TAPER:-1.5}"           # mild short-bias (flatter than 2.0, not uniform); 1.0=uniform
# curriculum OFF (stationary uniform distribution from step 0).
export NLA_TRUNC_CURRICULUM_GROUPS="${NLA_TRUNC_CURRICULUM_GROUPS:-0}"
# per-item length penalty (item 4a). 0 = off; try a small value once item-mode is verified.
export NLA_ITEM_LEN_PENALTY="${NLA_ITEM_LEN_PENALTY:-0}"
export NLA_ITEM_LEN_TARGET="${NLA_ITEM_LEN_TARGET:-25}"

# ───────────────────────── checkpoints (v2 warm-start outputs) ──────────────
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:?v2 warm-start AV: HF dir (list-format, NLA_NO_TRAIN_EOS warm-start)}"
: "${CRITIC_SL_CKPT:?v2 warm-start AR: HF dir with value_head.safetensors (item-truncated warm-start)}"
: "${RUN_DIR:=/workspace/rl_v2}"
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"

# ───────────────────────── KL ON by default (item 1) ───────────────────────
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"

# ───────────────────────── RL data (auto-built if missing) ──────────────────
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl.parquet}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet -> $RL_PARQUET ==="
    "${PYTHON:-python}" "$HERE/05_build_rl_parquet.py"
fi

# ───────────────────────── single-node 8×H100 profile ──────────────────────
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# Longer output room than v1's 150: item-mode wants headroom to produce many items
# before we post-truncate to K (and to let "infinite output" actually stretch).
ROLLOUT_MAX_RESP="${ROLLOUT_MAX_RESP:-256}"
ROLLOUT_MAX_CTX="${ROLLOUT_MAX_CTX:-512}"

# ───────────────────────── resume detection ────────────────────────────────
LATEST_ACTOR=""; LATEST_CRITIC=""
if compgen -G "$RUN_DIR/actor/iter_*" > /dev/null 2>&1; then
    LATEST_ACTOR="$(ls -d "$RUN_DIR"/actor/iter_* | sort | tail -1)"
    CRITIC_ITER="$(basename "$LATEST_ACTOR")"
    [ -d "$RUN_DIR/critic/$CRITIC_ITER/hf" ] && LATEST_CRITIC="$RUN_DIR/critic/$CRITIC_ITER/hf"
fi
if [ -n "$LATEST_ACTOR" ]; then
    export LOAD="$LATEST_ACTOR"
    [ -n "$LATEST_CRITIC" ] && export CRITIC_LOAD="$LATEST_CRITIC"
    echo "=== RESUMING from $LATEST_ACTOR (critic: ${LATEST_CRITIC:-<SFT>}) ==="
else
    export LOAD="${LOAD:-none}"
    echo "=== FRESH v2 RL start from warm-start HF checkpoints ==="
fi

# ───────────────────────── wandb (always on) ───────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /root/.wandb_key)}"
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-matryoshka-sonnet46-v2
            --wandb-team octahedral-systems
            --wandb-group "${WANDB_GROUP:-qwen2.5-7b-L20-rl-v2}" --wandb-mode online)

# ───────────────────────── dump exact config ───────────────────────────────
mkdir -p "$RUN_DIR"
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA v2 RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET KL_LOSS_COEF \
             NLA_TRUNC_MODE NLA_TRUNC_MAX_ITEMS NLA_TRUNC_TAPER NLA_TRUNC_CURRICULUM_GROUPS \
             NLA_ITEM_LEN_PENALTY NLA_ITEM_LEN_TARGET \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT ROLLOUT_MAX_RESP ROLLOUT_MAX_CTX \
             LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
echo "=== v2 RL start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
echo "    truncation: items mode, keep ~taper($NLA_TRUNC_TAPER) of [1, $NLA_TRUNC_MAX_ITEMS], KL=$KL_LOSS_COEF"
# rl.sh fixes 128×8=1024; trailing flags OVERRIDE to the single-node profile + the
# v2 longer-output caps (argparse: last value wins).
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESP" \
    --rollout-max-context-len "$ROLLOUT_MAX_CTX" \
    --sglang-context-length "$ROLLOUT_MAX_CTX" \
    "$@"
echo "=== v2 RL done $(date -u +%FT%TZ) ==="
