#!/bin/bash
# Continue RL on the released kitft Qwen2.5-7B L20 pair with the QUOTE-MARK
# PENALTY (see README.md). Thin wrapper over configs/rl.sh: v1 recipe (no
# truncation, 150-token cap, tagged format), single-node 512-batch profile,
# KL 0.01 anchored to the kitft AV, resume detection, launch-config dump.
#
# DEFAULT PROFILE: 150 steps on ONE 8×H100/H200 node (actor 4 / critic 2 /
# rollout 2), 512-sample batch (64 prompts × 8). ~45s/step ⇒ ~2h.
#   >>> COST CAP: provision the box at <= $20/hr total. <<<
set -euo pipefail

# ───────────────────────── the experiment knob ──────────────────────────────
# -0.1 per quotation-mark char (any kind, incl. apostrophes — see nla/reward.py
# _QUOTE_CHARS). Mild start: comparable to the ~0.1 within-group reward spread;
# raise if quoting doesn't budge.
export NLA_QUOTE_PENALTY="${NLA_QUOTE_PENALTY:-0.1}"
# Qualitative readout: reward.py overwrites this file with the first 20 samples
# of each reward batch; a poller should archive timestamped copies.
export NLA_ROLLOUT_TEXT_DUMP="${NLA_ROLLOUT_TEXT_DUMP:-/workspace/out/rollout_dump.txt}"

# ───────────────────────── checkpoints (released kitft pair) ────────────────
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:=/workspace/models/kitft_av}"   # kitft/nla-qwen2.5-7b-L20-av (HF dir, has nla_meta.yaml)
: "${CRITIC_SL_CKPT:=/workspace/models/kitft_ar}"   # kitft/nla-qwen2.5-7b-L20-ar (HF dir, has value_head.safetensors)
: "${RUN_DIR:=/workspace/rl_quotepen}"
# Fresh start: actor weights from --hf-checkpoint (the kitft AV), no DCP --load.
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"
# KL 0.01 to the kitft AV (rl.sh loads --ref-load = ACTOR_SFT_CKPT). The ref
# quotes heavily, so KL partially opposes the penalty — intentional: production
# default, keeps outputs coherent while the penalty pushes the marks out.
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"

# ───────────────────────── RL data (01_build_rl_parquet.py) ─────────────────
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl_tagged.parquet}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet -> $RL_PARQUET ==="
    "${PYTHON:-python}" "$HERE/01_build_rl_parquet.py"
fi

# ───────────────────────── single-node 8-GPU profile ────────────────────────
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
# 512-batch + √(512/1024)-scaled LR (production was 1024 @ 1.41e-5).
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_ROLLOUT="${NUM_ROLLOUT:-150}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

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
    echo "=== RESUMING from $LATEST_ACTOR (critic: ${LATEST_CRITIC:-<kitft AR>}) ==="
else
    export LOAD="${LOAD:-none}"
    echo "=== FRESH RL start from the kitft pair (actor weights via --hf-checkpoint) ==="
fi

# ───────────────────────── wandb (always on, per ~/ENV.md) ──────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /root/.wandb_key)}"
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-quote-penalty
            --wandb-team octahedral-systems
            --wandb-group "${WANDB_GROUP:-kitft-7b-L20-quotepen${NLA_QUOTE_PENALTY}}"
            --wandb-mode online)

# ───────────────────────── save-everything: dump exact config ───────────────
mkdir -p "$RUN_DIR" /workspace/out
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA quote-penalty RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET \
             NLA_QUOTE_PENALTY KL_LOSS_COEF NLA_ROLLOUT_TEXT_DUMP \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
# rl.sh runs `python train.py` relative to CWD — that's miles' entrypoint.
cd "${MILES_DIR:-/workspace/miles}"
echo "=== RL (quote penalty $NLA_QUOTE_PENALTY) start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    "$@"
echo "=== RL (quote penalty) done $(date -u +%FT%TZ) ==="
