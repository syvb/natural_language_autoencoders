#!/bin/bash
# v3qf ("quote-free"): CONTINUE RL on the v3 matryoshka Qwen2.5-7B L20 pair
# (syvb/nla-qwen2.5-7b-L20-v3-rl iter_0000200) with the QUOTE-MARK PENALTY
# (see README.md). The full v3 recipe is preserved — bullets format, uniform
# TOKEN truncation ~U[1,120], KL 0.03 — with one reward change:
#
#     reward = -mse_nrm - NLA_QUOTE_PENALTY * (# quote chars in explanation)
#
# Fresh RL start from the v3-final HF exports (actor weights via
# --hf-checkpoint, critic via --critic-load; optimizer state fresh). The KL
# reference is the v3-final AV itself — which quotes — so KL partially opposes
# the penalty; intentional, same tension as the kitft quote-penalty run.
#
# DEFAULT PROFILE: 150 steps on ONE 8×H100/H200 node (actor 4 / critic 2 /
# rollout 2), 512-sample batch (64 prompts × 8), resume-extendable.
#   >>> COST CAP: provision the box at <= $20/hr total. <<<
set -euo pipefail

# ───────────────────────── the experiment knob ──────────────────────────────
# -0.1 per quotation-mark char (any kind, incl. apostrophes — see nla/reward.py
# _QUOTE_CHARS). Strong by design: typical within-group |reward| spread ~0.1,
# and a quoting v3 bullet list carries several marks.
export NLA_QUOTE_PENALTY="${NLA_QUOTE_PENALTY:-0.1}"
# Qualitative readout: reward.py overwrites this file with the first 20 samples
# of each reward batch; a poller should archive timestamped copies.
export NLA_ROLLOUT_TEXT_DUMP="${NLA_ROLLOUT_TEXT_DUMP:-/workspace/out/rollout_dump.txt}"
# Quantitative source of truth: FULL-batch per-drain quote stats, JSONL.
export NLA_QUOTE_STATS_JSONL="${NLA_QUOTE_STATS_JSONL:-/workspace/out/quote_stats.jsonl}"

# ─────────────────── truncation: uniform TOKENS mode (v3) ───────────────────
export NLA_TRUNC_MODE="${NLA_TRUNC_MODE:-tokens}"
export NLA_TRUNC_MIN_TOKENS="${NLA_TRUNC_MIN_TOKENS:-1}"
export NLA_TRUNC_MAX_TOKENS="${NLA_TRUNC_MAX_TOKENS:-120}"
# Bullets actor emits no opening tag → offset 0 (see run_rl_v3.sh for why this
# is pinned rather than trusted to sidecar detection).
export NLA_TRUNC_OPENING_OFFSET="${NLA_TRUNC_OPENING_OFFSET:-0}"
# Zero the v2 item-mode knobs (lingering exports would flip mode/shape rewards).
export NLA_TRUNC_MAX_ITEMS=0 NLA_ITEM_LEN_PENALTY=0

# ──────────────── checkpoints (v3-final RL exports, iter_0000200) ───────────
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:=/workspace/models/v3_av}"   # syvb/nla-qwen2.5-7b-L20-v3-rl iter_0000200/av
: "${CRITIC_SL_CKPT:=/workspace/models/v3_ar}"   # syvb/nla-qwen2.5-7b-L20-v3-rl iter_0000200/ar (has value_head.safetensors)
: "${RUN_DIR:=/workspace/rl_v3qf}"
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"

# v3-final sidecar must carry the bullets template (guaranteed by
# push_v3_ckpt_to_hf.sh at upload, but a stale/wrong download dir would
# silently ship the tagged prompt).
if grep -q "<explanation>" "$ACTOR_SFT_CKPT/nla_meta.yaml" 2>/dev/null; then
    echo "FATAL: $ACTOR_SFT_CKPT/nla_meta.yaml has the v1 TAGGED actor template." >&2
    exit 1
fi

# ───────────────────────── KL 0.03 (v3 recipe) ──────────────────────────────
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.03}"

# ───────────────────────── RL data (01_build_rl_parquet.py) ─────────────────
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl_v3.parquet}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet (bullets) -> $RL_PARQUET ==="
    "${PYTHON:-python}" "$HERE/01_build_rl_parquet.py"
fi

# ───────────────────────── single-node 8-GPU profile ────────────────────────
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_ROLLOUT="${NUM_ROLLOUT:-150}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# Generation capped per group at the content budget (≤120 tokens, no opening
# tag) — 160 leaves slack.
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
    echo "=== RESUMING from $LATEST_ACTOR (critic: ${LATEST_CRITIC:-<v3-final AR>}) ==="
else
    export LOAD="${LOAD:-none}"
    echo "=== FRESH v3qf RL start from the v3-final pair (actor weights via --hf-checkpoint) ==="
fi

# ───────────────────────── wandb (always on, per ~/ENV.md) ──────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /root/.wandb_key)}"
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-quote-penalty
            --wandb-team octahedral-systems
            --wandb-group "${WANDB_GROUP:-qwen2.5-7b-L20-v3qf-quotepen${NLA_QUOTE_PENALTY}}"
            --wandb-mode online)

# ───────────────────────── save-everything: dump exact config ───────────────
mkdir -p "$RUN_DIR" /workspace/out
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA v3qf (quote-free) RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET \
             NLA_QUOTE_PENALTY NLA_QUOTE_STATS_JSONL NLA_ROLLOUT_TEXT_DUMP KL_LOSS_COEF \
             NLA_TRUNC_MODE NLA_TRUNC_MIN_TOKENS NLA_TRUNC_MAX_TOKENS \
             NLA_TRUNC_OPENING_OFFSET NLA_TRUNC_SEED NLA_TRUNC_MAX_ITEMS NLA_ITEM_LEN_PENALTY HF_CKPT \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT ROLLOUT_MAX_RESP ROLLOUT_MAX_CTX \
             LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
# rl.sh runs `python train.py` relative to CWD — that's miles' entrypoint.
cd "${MILES_DIR:-/workspace/miles}"
echo "=== v3qf RL (quote penalty $NLA_QUOTE_PENALTY) start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
echo "    truncation: tokens ~U[$NLA_TRUNC_MIN_TOKENS, $NLA_TRUNC_MAX_TOKENS], KL=$KL_LOSS_COEF"
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESP" \
    --rollout-max-context-len "$ROLLOUT_MAX_CTX" \
    --sglang-context-length "$ROLLOUT_MAX_CTX" \
    "$@"
echo "=== v3qf RL done $(date -u +%FT%TZ) ==="
