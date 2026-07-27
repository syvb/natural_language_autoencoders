#!/bin/bash
# Arm S ("standard objective") of the warm-start-vs-objective ablation
# (PLAN_v3std_objective_ablation.md). Identical to run_rl_v3.sh in every way
# except the truncation budget is CONSTANT: min == max == 120 content tokens,
# i.e. the reward is always computed on the full fixed-length output instead
# of a random-length prefix. Same code path as the matryoshka arm (at-cap
# TRUNCATED samples scored, no token-limit penalty) — required because the v3
# AV never emits EOS, so truncation-off would collapse every rollout to the
# failed-extraction penalty.
#
# The three knobs that define the ablation are HARD-PINNED (not ${VAR:-}
# fallbacks): a lingering export from a reused shell must not be able to turn
# this back into the matryoshka arm or move the KL off the shared 0.03.
set -euo pipefail

export NLA_TRUNC_MODE=tokens
export NLA_TRUNC_MIN_TOKENS=120
export NLA_TRUNC_MAX_TOKENS=120
export KL_LOSS_COEF=0.03          # matches arm M (the existing v3 RL run); constant, no decay

export RUN_DIR="${RUN_DIR:-/workspace/rl_v3std}"
export WANDB_GROUP="${WANDB_GROUP:-qwen2.5-7b-L20-rl-v3std}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_rl_v3.sh" "$@"
