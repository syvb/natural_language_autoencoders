#!/bin/bash
# Step 7 — matryoshka RL: uniform token truncation + position-tapered KL.
# Thin wrapper over configs/rl.sh (miles train.py). Recipe = the validated 7B
# v3 recipe with one change:
#
#   TAPERED KL — --kl-loss-coef ($KL_LOSS_COEF, default 0.02) applies at the
#   FIRST response token and halves every NLA_KL_TAPER_HALF_LIFE tokens
#   (nla/kl_taper.py). Rationale: truncation front-loads information; late
#   tokens contribute little FVE and were dominated by a flat KL. Watch
#   train/kl_loss (weighted, in the loss) vs train/kl_flat (unweighted).
#
# Truncation: content-token budget ~U[MIN, MAX] shared per GRPO group, applied
# as a max_new_tokens cap (no post-truncation). min=1 is safe: the AR
# warm-start was pre-calibrated on the same U-draw (02) and the grad guard
# skips any non-finite-grad step.
#
# GPU layout: ACTOR_GPUS + CRITIC_GPUS + ROLLOUT_GPUS on one node — three
# DISJOINT pools. 27B on 8x B200-192GB: 4/2/2. On H200-141GB the critic may
# not fit critic2 — fall back to 4/3/1 (see RUNBOOK).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# ─────────────────── truncation: uniform TOKENS mode ─────────────────────────
export NLA_TRUNC_MODE="${NLA_TRUNC_MODE:-tokens}"
export NLA_TRUNC_MIN_TOKENS NLA_TRUNC_MAX_TOKENS
# The bullets actor emits no opening tag; pin the offset so a stale tagged
# sidecar can't silently shift the budget window.
export NLA_TRUNC_OPENING_OFFSET="${NLA_TRUNC_OPENING_OFFSET:-0}"
export NLA_TRUNC_MAX_ITEMS=0 NLA_ITEM_LEN_PENALTY=0

# ─────────────────── tapered KL (this run's new lever) ───────────────────────
export KL_LOSS_COEF NLA_KL_TAPER_HALF_LIFE NLA_KL_TAPER_FLOOR

# ───────────────────── checkpoints (warm-start outputs) ──────────────────────
: "${ACTOR_SFT_CKPT:?warm-start AV HF dir (04_convert_upload.py -> \$WORK/hf_out/av_ws)}"
: "${CRITIC_SL_CKPT:?warm-start AR HF dir with value_head.safetensors (\$WORK/ckpt/ar_ws/iter_*/hf)}"
: "${INSTRUCT_MODEL:=${BASE_MODEL_DIR:-$WORK/models/base}}"
: "${RUN_DIR:=$WORK/rl_run}"
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"

[ -f "$ACTOR_SFT_CKPT/nla_meta.yaml" ] || {
    echo "FATAL: $ACTOR_SFT_CKPT/nla_meta.yaml missing — not a converted warm-start" >&2
    echo "(run 04_convert_upload.py first; also check \$WORK is exported in your shell)" >&2
    exit 1
}
if grep -q "<explanation>" "$ACTOR_SFT_CKPT/nla_meta.yaml"; then
    echo "FATAL: $ACTOR_SFT_CKPT/nla_meta.yaml has the TAGGED actor template." >&2
    echo "Re-export the warm-start with --nla-sidecar-source = av_sft.parquet." >&2
    exit 1
fi

# ───────────────────────── RL data (auto-built if missing) ───────────────────
export RL_PARQUET="${RL_PARQUET:-$WORK/out/rl.parquet}"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet -> $RL_PARQUET ==="
    (cd "$HERE" && "${PYTHON:-python}" "$HERE/05_build_rl_parquet.py")
fi

# ───────────────────────── GPU / optimizer profile ───────────────────────────
export ACTOR_NODES=1 CRITIC_NODES=1
export ACTOR_GPUS CRITIC_GPUS ROLLOUT_GPUS ACTOR_LR CRITIC_LR SAVE_INTERVAL
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# ───────────────────────── resume detection ──────────────────────────────────
# miles resolves --load as the save ROOT (reads latest_checkpointed_iteration.txt
# then iter_%07d/). Passing an iter_* dir makes miles_validate_args silently
# fall back to --ref-load at rollout 0 — a "resume" that restarts from scratch.
TRACKER="$RUN_DIR/actor/latest_checkpointed_iteration.txt"
if [ -f "$TRACKER" ]; then
    export LOAD="$RUN_DIR/actor"
    LATEST_IT="$(cat "$TRACKER")"
    CRITIC_HF="$RUN_DIR/critic/iter_$(printf %07d "$LATEST_IT")/hf"
    [ -d "$CRITIC_HF" ] && export CRITIC_LOAD="$CRITIC_HF"
    echo "=== RESUMING from $LOAD @ iter $LATEST_IT (critic: ${CRITIC_LOAD:-<SFT>}) ==="
else
    export LOAD="${LOAD:-none}"
    echo "=== FRESH RL start from warm-start HF checkpoints ==="
fi

# ───────────────────────── wandb (always on) ─────────────────────────────────
if [ -z "${WANDB_API_KEY:-}" ]; then
    [ -f "$WANDB_KEY_FILE" ] || { echo "FATAL: $WANDB_KEY_FILE missing (wandb is mandatory)" >&2; exit 1; }
    WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi
export WANDB_API_KEY
WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT"
            --wandb-team "$WANDB_TEAM"
            --wandb-group "${WANDB_GROUP:-${MODEL_TAG}-rl}" --wandb-mode online)

# ───────────────────────── dump exact config ─────────────────────────────────
mkdir -p "$RUN_DIR"
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in BASE_MODEL MODEL_TAG LAYER_INDEX INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT \
             RUN_DIR RL_PARQUET KL_LOSS_COEF NLA_KL_TAPER_HALF_LIFE NLA_KL_TAPER_FLOOR \
             NLA_TRUNC_MODE NLA_TRUNC_MIN_TOKENS NLA_TRUNC_MAX_TOKENS \
             NLA_TRUNC_OPENING_OFFSET NLA_TRUNC_SEED HF_CKPT \
             ACTOR_GPUS CRITIC_GPUS ROLLOUT_GPUS ACTOR_LR CRITIC_LR \
             SAVE_INTERVAL NUM_ROLLOUT RL_ROLLOUT_BS RL_GLOBAL_BS \
             ROLLOUT_MAX_RESP ROLLOUT_MAX_CTX LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
echo "=== RL start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
echo "    truncation ~U[$NLA_TRUNC_MIN_TOKENS,$NLA_TRUNC_MAX_TOKENS] tokens; KL $KL_LOSS_COEF @ t=0, half-life $NLA_KL_TAPER_HALF_LIFE tokens"
# rl.sh runs `python train.py` from cwd — it must be the miles checkout (also
# keeps cwd clear of the parent-of-repo namespace-shadow trap, see 03).
cd "${MILES_DIR:-$WORK/miles}"
# Extra sglang server flags, space-separated (word-split on purpose). miles
# mirrors the INSTALLED sglang's ServerArgs with a --sglang- prefix, so
# version-specific flags go here rather than hardcoded. Known need:
# sglang >=0.5.10 enables piecewise CUDA graphs by default and its warmup
# compile crashed with an illegal memory access on Hopper (0.5.10 smoke) —
# set SGLANG_EXTRA_ARGS="--sglang-disable-piecewise-cuda-graph" there.
if [ -z "${SGLANG_EXTRA_ARGS+x}" ]; then
    SGLANG_EXTRA_ARGS="$(python - <<'PYEOF'
import sglang
maj, mid, *_ = sglang.__version__.split(".")
mid = int("".join(c for c in mid if c.isdigit()) or 0)
patch = sglang.__version__.split(".")[2] if sglang.__version__.count(".") >= 2 else "0"
patch = int("".join(c for c in patch if c.isdigit()) or 0)
print("--sglang-disable-piecewise-cuda-graph" if (int(maj), mid, patch) >= (0, 5, 10) else "")
PYEOF
)"
fi
# rl.sh defaults 128x8=1024; trailing flags override (argparse last-wins).
# shellcheck disable=SC2086
bash "$REPO_ROOT/configs/rl.sh" \
    $SGLANG_EXTRA_ARGS \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size "$RL_ROLLOUT_BS" --global-batch-size "$RL_GLOBAL_BS" \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESP" \
    --rollout-max-context-len "$ROLLOUT_MAX_CTX" \
    --sglang-context-length "$ROLLOUT_MAX_CTX" \
    "$@"
echo "=== RL done $(date -u +%FT%TZ) ==="
