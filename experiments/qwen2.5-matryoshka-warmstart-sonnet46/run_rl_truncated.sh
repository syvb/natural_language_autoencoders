#!/bin/bash
# RL the matryoshka-warm-started Qwen2.5-7B L20 AV/AR pair with RANDOM-LENGTH
# TRUNCATION ("information-upfront"). Thin wrapper over configs/rl.sh: adds the
# truncation env vars, a single-node 512-batch profile, the RL-parquet build,
# resume detection, and a launch-config dump. All RL plumbing lives in rl.sh.
#
# DEFAULT PROFILE: a 300-step signal run on ONE 8×H100 node (actor 4 / critic 2
# / rollout 2), 512-sample batch (64 prompts × 8). ~45s/step ⇒ ~3.75h.
#
#   >>> COST CAP: provision the 8×H100 box at <= $20/hr TOTAL (<= $2.50/GPU/hr).
#       On vast.ai bid accordingly; a 300-step run is ~3.75h ⇒ ~$75 at $20/hr. <<<
#
# What truncation does (see nla/truncation.py for the full rationale):
#   - each AV generation is capped at a random number of explanation CONTENT
#     tokens, uniform in [MIN, MAX] and SHARED across a prompt's 8 samples
#     (clean GRPO within-group comparison),
#   - capping (not post-truncation) length-limits the trained tokens/loss-mask/
#     logprobs with no token surgery,
#   - the closing </explanation> tag is never trained/rewarded,
#   - the old token-limit penalty is removed (hitting the random cap is normal).
set -euo pipefail

# ───────────────────────── truncation (turns the feature ON) ────────────────
export NLA_TRUNC_MIN_TOKENS="${NLA_TRUNC_MIN_TOKENS:-1}"
export NLA_TRUNC_MAX_TOKENS="${NLA_TRUNC_MAX_TOKENS:-130}"
# export NLA_TRUNC_SEED=...        # optional; defaults to --rollout-seed

# ───────────────────────── checkpoints (warm-start outputs) ─────────────────
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:?DCP iter dir from the AV warm-start, e.g. /workspace/ckpt/av/iter_0000859}"
: "${CRITIC_SL_CKPT:?HF dir from the AR warm-start, e.g. /workspace/ckpt/ar/iter_0000857/hf}"
: "${RUN_DIR:=/workspace/rl_trunc}"

# ───────────────────────── RL data (auto-built if missing) ──────────────────
# stage-"rl" parquet: prompts + activation_vector from the AV TRAIN split
# (doc-disjoint from av_eval). 05_build_rl_parquet.py is idempotent.
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl.parquet}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet -> $RL_PARQUET ==="
    "${PYTHON:-python}" "$HERE/05_build_rl_parquet.py"
fi

# ───────────────────────── single-node 8×H100 profile ──────────────────────
# Pools are DISJOINT in miles' placement group: 4 + 2 + 2 = 8 GPUs.
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
# 512-batch + √(512/1024)-scaled LR (production was 1024 @ 1.41e-5). Parity LRs.
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"      # 6 checkpoints over 300 steps
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
# embedding dump to tmpfs (RAM), not overlay disk — ~1.5s→0.1s per step
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# ───────────────────────── resume detection ────────────────────────────────
# If $RUN_DIR/actor already has saved iters, CONTINUE from the latest (weights +
# optimizer + rollout_id). REF stays the SFT (fixed KL reference). Otherwise a
# fresh RL start from the warm-start checkpoints.
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
    echo "=== FRESH RL start from warm-start checkpoints ==="
fi

# ───────────────────────── wandb (always on, per ~/ENV.md) ──────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /root/.wandb_key)}"
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-matryoshka-sonnet46-trunc
            --wandb-team octahedral-systems)

# ───────────────────────── save-everything: dump exact config ───────────────
mkdir -p "$RUN_DIR"
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA truncated-RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET \
             NLA_TRUNC_MIN_TOKENS NLA_TRUNC_MAX_TOKENS NLA_TRUNC_SEED \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
echo "=== RL (truncated) start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
echo "    truncation: content tokens ~U[$NLA_TRUNC_MIN_TOKENS, $NLA_TRUNC_MAX_TOKENS], shared per group"
echo "    GPUs: actor=$ACTOR_GPUS critic=$CRITIC_GPUS rollout=$ROLLOUT_GPUS (one node, <=\$20/hr)"
# rl.sh fixes 128×8=1024; the trailing flags OVERRIDE to the single-node profile
# (argparse: last value wins). --num-rollout is not set in rl.sh, so we add it.
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    "$@"
echo "=== RL (truncated) done $(date -u +%FT%TZ) ==="
