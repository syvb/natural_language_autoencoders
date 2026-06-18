#!/bin/bash
# KL-anchored length-penalty sweep + per-penalty held-out generation.
# Per penalty: train STEPS (KL vs released AV; saves actor DCP + critic HF)
#   -> held-out eval (load trained AV via --load) -> extract samples.jsonl
#   -> drop big checkpoints (keep train.log/rollout metrics + held-out samples).
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate miles
SEED="${SEED:-1234}"; STEPS="${STEPS:-35}"; KL="${KL_COEF:-0.01}"
PENALTIES="${PENALTIES:-0.006 0.004 0.002 0.001 0}"
: "${WANDB_KEY:?set WANDB_KEY}"; export WANDB_KEY
ITER=$(printf "iter_%07d" "$STEPS")

clean() { ray stop --force >/dev/null 2>&1 || true; pkill -9 -f sglang 2>/dev/null || true; rm -rf /dev/shm/nla/* 2>/dev/null || true; sleep 8; }

echo "######## KL SWEEP START $(date) seed=$SEED steps=$STEPS kl=$KL penalties='$PENALTIES' ########"
for p in $PENALTIES; do
  tag="pen${p}_kl_seed${SEED}"
  echo "######## TRAIN penalty=$p tag=$tag $(date) ########"
  KL_COEF="$KL" PENALTY="$p" SEED="$SEED" STEPS="$STEPS" TAG="$tag" SAVE_INTERVAL="$STEPS" \
    WANDB_KEY="$WANDB_KEY" bash /workspace/rl_run.sh || echo "!! TRAIN FAILED $tag"
  clean
  actor_dcp="/workspace/results/$tag/actor"               # save dir (has latest_checkpointed_iteration.txt for --load)
  critic_hf="/workspace/results/$tag/critic/$ITER/hf"
  if [ -d "$actor_dcp/$ITER" ] && [ -d "$critic_hf" ]; then
    echo "######## HELD-OUT EVAL $tag $(date) ########"
    ACTOR_DCP="$actor_dcp" CRITIC_HF="$critic_hf" TAG="${tag}_eval" bash /workspace/eval_heldout.sh || echo "!! EVAL FAILED $tag"
    clean
    python /workspace/extract_heldout.py "${tag}_eval" || echo "!! EXTRACT FAILED $tag"
  else
    echo "!! MISSING CKPT $tag (actor=$actor_dcp critic=$critic_hf) — skipping held-out"
  fi
  # reclaim disk: drop 84GB actor DCP + critic weights; keep metrics + held-out samples
  rm -rf "/workspace/results/$tag/actor" "/workspace/results/$tag/critic"
  echo "######## DONE $tag $(date) ########"
done
echo "######## KL SWEEP COMPLETE $(date) ########"
