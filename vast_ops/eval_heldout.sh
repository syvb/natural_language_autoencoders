#!/bin/bash
# Held-out eval of a saved AV+AR (one penalty's trained model): 1 rollout, penalty
# OFF (reward = pure -MSE), over the held-out set -> dump ~1024 (explanation,
# reconstruction-MSE) samples. Reuses the RL rollout path; no separate inference stack.
# Env: AV (saved AV hf dir), AR (saved AR hf dir), TAG.
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate miles
export PYTHONPATH=/workspace/nla:${PYTHONPATH:-}
export NLA_EMBED_DUMP_DIR=/dev/shm/nla; mkdir -p "$NLA_EMBED_DUMP_DIR"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
NGPU=$(nvidia-smi -L | wc -l); export CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NGPU)))")
export NLA_LENGTH_PENALTY=0   # eval: reward is pure -MSE (reconstruction only)

# ACTOR_DCP = trained actor DCP iter dir (actor saves DCP-only); base arch from the
# released AV. CRITIC_HF = trained critic HF dir (critic saves HF + value_head).
# NOTE: do NOT name these AR/AS/CC/etc — conda activate exports those (compiler
# tools), clobbering them. (AR -> x86_64-conda-linux-gnu-ar.)
ACTOR_DCP="${ACTOR_DCP:?set ACTOR_DCP (trained actor DCP save dir, or BASE for released AV)}"; CRITIC_HF="${CRITIC_HF:?set CRITIC_HF (trained/released critic hf dir)}"; TAG="${TAG:?set TAG}"
# BASE = evaluate the released NLA itself (no continued-RL weights). Else overlay trained AV.
LOAD_FLAGS=()
[ "$ACTOR_DCP" != "BASE" ] && LOAD_FLAGS=(--load "$ACTOR_DCP" --finetune)
OUT="/workspace/heldout_results/$TAG"; mkdir -p "$OUT/rollout"
cd /workspace/miles
python train.py \
  --train-backend fsdp --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
  --loss-type policy_loss --advantage-estimator grpo --force-use-critic \
  --n-samples-per-prompt 2 \
  --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
  --custom-generate-function-path nla.rollout.nla_generate.generate \
  --custom-rm-path nla.reward.nla_rm --data-source-path nla.data_source.NLADataSource \
  --prompt-data /workspace/heldout/heldout.parquet --input-key prompt \
  --hf-checkpoint /workspace/models/av "${LOAD_FLAGS[@]}" \
  --nla-sidecar-source /workspace/models/av \
  --critic-load "$CRITIC_HF" --nla-critic-sidecar-source "$CRITIC_HF" \
  --critic-save "/workspace/evaltmp_$TAG/critic" \
  --save-debug-rollout-data "$OUT/rollout/step_{rollout_id}.pt" \
  --lr 1.41e-5 --critic-lr 1.41e-5 --lr-decay-style constant \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --critic-num-nodes 1 \
  --critic-num-gpus-per-node 1 --rollout-num-gpus 1 \
  --rollout-max-response-len 150 --rollout-max-context-len 300 \
  --sglang-disable-radix-cache --sglang-context-length 300 \
  --router-history-backend none --router-policy round_robin --router-disable-circuit-breaker \
  --router-retry-max-backoff-ms 500 --router-retry-max-retries 2 \
  --rollout-batch-size 512 --global-batch-size 1024 --micro-batch-size 4 \
  --num-rollout 1 --loss-mask-type qwen --rollout-seed 9999 \
  2>&1 | tee "$OUT/eval.log"
rm -rf "/workspace/evaltmp_$TAG" 2>/dev/null   # drop any throwaway checkpoint save
echo "=== eval $TAG done $(date) ==="
