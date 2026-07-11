#!/bin/bash
# Continue RL on the released kitft Qwen2.5-7B L20 pair with the COPIED-BITS
# penalty (see README.md + the quote-penalty postmortem): charge λ per bit of
# input text reproduced verbatim beyond the chance-collision deductible.
# Thin wrapper over configs/rl.sh: v1 recipe (no truncation, 150-token cap,
# tagged format), KL 0.01 anchored to the kitft AV, resume detection.
#
# DEFAULT PROFILE: 100 steps on ONE 8×H100 node (actor 4 / critic 2 /
# rollout 2), 512-sample batch (64 prompts × 8). ~31s/step ⇒ ~52 min.
set -euo pipefail

# ───────────────────────── the experiment knobs ─────────────────────────────
# λ = 0.01/bit: step-0 mean charged bits ≈ 19 (offline calibration) ⇒ mean
# penalty ≈ 0.19, ~2× the within-group reward spread. ~6× gentler than the
# quote penalty was; raise via resume if copying doesn't move by ~step 50.
export NLA_OVERLAP_PENALTY="${NLA_OVERLAP_PENALTY:-0.01}"
export NLA_OVERLAP_DEDUCTIBLE="${NLA_OVERLAP_DEDUCTIBLE:-15}"
export NLA_OVERLAP_MIN_SPAN="${NLA_OVERLAP_MIN_SPAN:-2}"
export NLA_OVERLAP_UNIGRAMS="${NLA_OVERLAP_UNIGRAMS:-/workspace/out/unigrams.json}"
# other shaping OFF, explicitly — a lingering export must not stack penalties.
export NLA_QUOTE_PENALTY=0
export NLA_REPEAT_PENALTY=0
# full-batch metrics: reward.py appends per-drain quote + overlap-bits stats;
# quote_stats_wandb.py reads this and logs to wandb next to the run.
export NLA_QUOTE_STATS_JSONL="${NLA_QUOTE_STATS_JSONL:-/workspace/out/quote_stats.jsonl}"
export NLA_ROLLOUT_TEXT_DUMP="${NLA_ROLLOUT_TEXT_DUMP:-/workspace/out/rollout_dump.txt}"

# ───────────────────────── checkpoints (released kitft pair) ────────────────
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:=/workspace/models/kitft_av}"
: "${CRITIC_SL_CKPT:=/workspace/models/kitft_ar}"
: "${RUN_DIR:=/workspace/rl_overlappen}"
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"

# ───────────────────────── RL data (prebuilt on the dev box) ────────────────
# Parquet carries detokenized_text_truncated (--keep-debug-metadata) — the
# copied-bits scorer matches against it. Built + verified by
# 01_build_rl_parquet_ctx.py; downloaded here, never built on the box.
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl_tagged_ctx.parquet}"
DATA_REPO="${DATA_REPO:-syvb/nla-qwen2.5-7b-L20-rl-overlappen}"
if [ ! -f "$RL_PARQUET" ] || [ ! -f "$NLA_OVERLAP_UNIGRAMS" ]; then
    echo "=== downloading prebuilt RL data from $DATA_REPO ==="
    python - "$DATA_REPO" <<'PY'
import sys
from huggingface_hub import hf_hub_download
tok = open("/root/.hf_token").read().strip()
for f in ("data/rl_tagged_ctx.parquet", "data/rl_tagged_ctx.parquet.nla_meta.yaml",
          "data/unigrams.json"):
    hf_hub_download(sys.argv[1], f, token=tok, local_dir="/workspace/_data")
PY
    mkdir -p /workspace/out
    cp /workspace/_data/data/rl_tagged_ctx.parquet* /workspace/out/
    cp /workspace/_data/data/unigrams.json /workspace/out/
fi

# ───────────────────────── single-node 8-GPU profile ────────────────────────
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# ───────────────────────── resume detection ────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
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
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-overlap-penalty
            --wandb-team octahedral-systems
            --wandb-group "${WANDB_GROUP:-kitft-7b-L20-overlappen${NLA_OVERLAP_PENALTY}}"
            --wandb-mode online)

# ───────────────────────── save-everything: dump exact config ───────────────
mkdir -p "$RUN_DIR" /workspace/out
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA copied-bits (overlap) RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET \
             NLA_OVERLAP_PENALTY NLA_OVERLAP_DEDUCTIBLE NLA_OVERLAP_MIN_SPAN \
             NLA_OVERLAP_UNIGRAMS NLA_QUOTE_PENALTY NLA_REPEAT_PENALTY \
             NLA_QUOTE_STATS_JSONL KL_LOSS_COEF \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
# rl.sh runs `python train.py` relative to CWD — that's miles' entrypoint.
cd "${MILES_DIR:-/workspace/miles}"
echo "=== RL (copied-bits λ=$NLA_OVERLAP_PENALTY, B0=$NLA_OVERLAP_DEDUCTIBLE) start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps ==="
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    "$@"
echo "=== RL (copied-bits) done $(date -u +%FT%TZ) ==="
