#!/bin/bash
# Length-penalty sweep: control (penalty=0) + penalties, all from the same released
# AV/AR checkpoint with the SAME seed. Sequential (each run uses all 4 GPUs).
# Metrics -> wandb (cloud) + per-step rollout dumps under /workspace/results/<tag>/.
set -uo pipefail
SEED="${SEED:-1234}"
STEPS="${STEPS:-50}"
PENALTIES="${PENALTIES:-0 0.003 0.01 0.03}"
: "${WANDB_KEY:?set WANDB_KEY}"
export WANDB_KEY
export WANDB_GROUP="sweep-seed${SEED}"

echo "===== SWEEP START $(date) seed=$SEED steps=$STEPS penalties='$PENALTIES' ====="
for p in $PENALTIES; do
  tag="pen${p}_seed${SEED}"
  echo "===== RUN penalty=$p tag=$tag $(date) ====="
  PENALTY="$p" SEED="$SEED" STEPS="$STEPS" TAG="$tag" \
    WANDB_KEY="$WANDB_KEY" WANDB_GROUP="$WANDB_GROUP" \
    bash /workspace/rl_run.sh || echo "!!!! RUN $tag FAILED (continuing) !!!!"
  # Tear down any lingering Ray cluster + free GPUs before the next run.
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f sglang >/dev/null 2>&1 || true
  sleep 10
done
echo "===== SWEEP DONE $(date) ====="
