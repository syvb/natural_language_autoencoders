#!/bin/bash
# RL the v3 bullets-format Qwen2.5-7B L20 AV/AR pair. Thin wrapper over
# configs/rl.sh, same shape as run_rl_v2.sh but with the v3 changes:
#
#   1. KL penalty 0.03 (v1: 0.01, v2: 0.02) — anchors the AV harder to the
#      warm-start reference.
#   2. Back to UNIFORM TOKEN truncation (v1 mechanism), but over the full
#      [1, 120] range instead of v1's [16, 150]: content-token budget ~U[1,120]
#      shared per group, applied as a max_new_tokens cap (no post-truncation).
#      min=1 is safe now because (a) the v3 AR warm-start was pre-calibrated on
#      U[1,120]-token prefixes (02c --ar-truncate-max-tokens) and (b) the
#      grad-finiteness guard skips any critic step with non-finite grads.
#      No item-length penalty — token truncation can't be gamed by cramming.
#   3. Bullets prompt everywhere: the RL parquet is built with
#      EXPLANATION_FORMAT=bullets so the rollout prompt (and the sidecar shipped
#      with every checkpoint) matches the warm-start's actual prompt. v2 shipped
#      with the old tagged "2-3 text snippets" prompt here — fixed.
#
# The AV emits a raw "- " bullet list (no <explanation> wrapper, never trained
# to stop), so extract_explanation_open reads the whole output and the tokens-
# mode opening offset auto-resolves to 0 (no tag in the sidecar template).
#
# DEFAULT PROFILE: matches run_rl_v2.sh — one 8×H100 node (actor 4 / critic 2 /
# rollout 2), 512-sample batch (64×8). Set NUM_ROLLOUT small for a smoke.
set -euo pipefail

# ─────────────────── truncation: uniform TOKENS mode (v3) ───────────────────
export NLA_TRUNC_MODE="${NLA_TRUNC_MODE:-tokens}"
export NLA_TRUNC_MIN_TOKENS="${NLA_TRUNC_MIN_TOKENS:-1}"
export NLA_TRUNC_MAX_TOKENS="${NLA_TRUNC_MAX_TOKENS:-120}"

# ───────────────────── checkpoints (v3 warm-start outputs) ──────────────────
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:?v3 warm-start AV: HF dir (bullets-format, NLA_NO_TRAIN_EOS warm-start)}"
: "${CRITIC_SL_CKPT:?v3 warm-start AR: HF dir with value_head.safetensors (token-truncated warm-start)}"
: "${RUN_DIR:=/workspace/rl_v3}"
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"

# ───────────────────────── KL 0.03 (v3 item 1) ──────────────────────────────
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.03}"

# ───────────────────────── RL data (auto-built if missing) ──────────────────
# NOT /workspace/out/rl.parquet — that name is the v1/v2 TAGGED-prompt parquet;
# a stale one on a reused box would silently give v3 the wrong prompt again.
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl_v3.parquet}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet (bullets) -> $RL_PARQUET ==="
    EXPLANATION_FORMAT=bullets "${PYTHON:-python}" "$HERE/05_build_rl_parquet.py"
fi

# ───────────────────────── single-node 8×H100 profile ──────────────────────
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# Generation is capped per group at the content budget (≤120 tokens, no opening
# tag), so 160 leaves slack without wasting KV cache on v2's 256.
ROLLOUT_MAX_RESP="${ROLLOUT_MAX_RESP:-160}"
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
    echo "=== FRESH v3 RL start from warm-start HF checkpoints ==="
fi

# ───────────────────────── wandb (always on) ───────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /root/.wandb_key)}"
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-matryoshka-sonnet46-v3
            --wandb-team octahedral-systems
            --wandb-group "${WANDB_GROUP:-qwen2.5-7b-L20-rl-v3}" --wandb-mode online)

# ───────────────────────── dump exact config ───────────────────────────────
mkdir -p "$RUN_DIR"
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA v3 RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET KL_LOSS_COEF \
             NLA_TRUNC_MODE NLA_TRUNC_MIN_TOKENS NLA_TRUNC_MAX_TOKENS \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT ROLLOUT_MAX_RESP ROLLOUT_MAX_CTX \
             LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
echo "=== v3 RL start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
echo "    truncation: tokens mode, ~U[$NLA_TRUNC_MIN_TOKENS, $NLA_TRUNC_MAX_TOKENS] content tokens, KL=$KL_LOSS_COEF"
# rl.sh fixes 128×8=1024; trailing flags OVERRIDE to the single-node profile + the
# v3 output caps (argparse: last value wins).
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESP" \
    --rollout-max-context-len "$ROLLOUT_MAX_CTX" \
    --sglang-context-length "$ROLLOUT_MAX_CTX" \
    "$@"
echo "=== v3 RL done $(date -u +%FT%TZ) ==="
